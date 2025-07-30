import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp


class PathDiscovery(nn.Module):
    """Module for discovering potential paths using random walks instead of BFS"""

    def __init__(self, node_num, hidden_dim=64, max_paths=8, max_path_length=5):
        super().__init__()
        self.max_paths = max_paths
        self.max_path_length = max_path_length
        self.node_num = node_num

        # Path feature encoder
        self.path_encoder = nn.GRU(
            hidden_dim, hidden_dim, bidirectional=True, batch_first=True
        )

        # Path importance scoring network
        self.path_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Path source point prediction
        self.source_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        # Gumbel temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def _guided_random_walk(
        self, adj_matrix, source, node_features, accident_indicators=None, alpha=0.15
    ):
        """
        Guided random walk based on adjacency matrix and accident information
        """
        walk_length = min(self.max_path_length, self.node_num)
        path = torch.full(
            (walk_length,), source.item(), dtype=torch.long, device=adj_matrix.device
        )
        visited = torch.zeros(self.node_num, dtype=torch.bool, device=adj_matrix.device)
        visited[source.item()] = True

        current_node = source.item()
        path_len = 1

        # Calculate feature differences for guidance
        current_feat = node_features[current_node]
        feature_diff = torch.norm(node_features - current_feat, dim=-1)
        feature_guidance = feature_diff / feature_diff.max() if feature_diff.max() > 0 else torch.zeros_like(feature_diff)

        # Execute guided random walk
        for i in range(walk_length - 1):
            transition_probs = adj_matrix[current_node].clone()

            # Apply restart probability
            if torch.rand(1, device=adj_matrix.device).item() < alpha:
                next_node = source.item()
            else:
                # Filter visited nodes
                transition_probs[visited] = 0.0
                if transition_probs.sum() == 0:
                    break

                # Integrate accident information into transition probabilities
                if accident_indicators is not None:
                    accident_bias = accident_indicators / (accident_indicators.sum() + 1e-8)
                    transition_probs = transition_probs * (1 + accident_bias)

                # Integrate feature guidance - bias towards nodes with larger feature differences
                transition_probs = transition_probs * (1 + feature_guidance)

                # Normalize probabilities
                if transition_probs.sum() > 0:
                    transition_probs = transition_probs / transition_probs.sum()
                else:
                    break

                # Sample next node
                next_node_dist = torch.distributions.Categorical(probs=transition_probs)
                next_node = next_node_dist.sample().item()

            # Update path and visited nodes
            path[path_len] = next_node
            visited[next_node] = True
            current_node = next_node
            path_len += 1

            # Update feature guidance
            current_feat = node_features[current_node]
            feature_diff = torch.norm(node_features - current_feat, dim=-1)
            feature_guidance = feature_diff / feature_diff.max() if feature_diff.max() > 0 else torch.zeros_like(feature_diff)

        return path

    def _structural_path_sampling(self, node_features, adj_matrix, accident_mask=None):
        """Sample potential paths based on graph structure and accident information"""
        batch_size, seq_len, num_nodes, feat_dim = node_features.shape

        # Find potential accident nodes as path starting points
        if accident_mask is not None:
            source_probs = accident_mask.float()
            accident_indicators = accident_mask.float()
        else:
            feature_diff = torch.abs(node_features[:, 1:] - node_features[:, :-1])
            feature_change = torch.mean(feature_diff, dim=(1, 3))
            source_logits = self.source_predictor(node_features[:, -1])
            source_probs = F.softmax(source_logits.squeeze(-1) * feature_change, dim=-1)
            accident_indicators = feature_change

        # Get possible path starting points
        k = min(self.max_paths, num_nodes)
        if self.training:
            gumbel_sample = F.gumbel_softmax(
                source_probs.log(), tau=self.temperature, hard=True
            )
            source_indices = gumbel_sample.topk(k, dim=-1)[1]
        else:
            source_indices = source_probs.topk(k, dim=-1)[1]

        # Generate random walk paths from each starting point
        collected_paths = []
        path_batch_indices = []

        for b in range(batch_size):
            for idx in range(source_indices.size(1)):
                source_idx = source_indices[b, idx]
                path = self._guided_random_walk(
                    adj_matrix[b],
                    source_idx,
                    node_features[b, -1],
                    accident_indicators=accident_indicators[b] if accident_indicators is not None else None,
                )
                collected_paths.append(path)
                path_batch_indices.append(b)

        # If no paths were sampled, return empty list and empty tensor
        if not collected_paths:
            empty_features = torch.empty(
                (
                    batch_size,
                    k,
                    self.path_encoder.hidden_size * (2 if self.path_encoder.bidirectional else 1),
                ),
                device=node_features.device,
            )
            return [[] for _ in range(batch_size)], empty_features

        # Batch extract path features
        paths_tensor = torch.stack(collected_paths)
        batch_indices_tensor = torch.tensor(
            path_batch_indices, dtype=torch.long, device=node_features.device
        )

        all_path_features_flat = self._extract_path_features_batched(
            paths_tensor, node_features, batch_indices_tensor
        )

        # Reshape output
        hidden_dim_out = all_path_features_flat.size(-1)
        all_path_features = all_path_features_flat.view(batch_size, k, hidden_dim_out)

        # Reorganize path list
        all_paths_grouped = [[] for _ in range(batch_size)]
        path_counter = 0
        for b in range(batch_size):
            for _ in range(k):
                if path_counter < len(collected_paths):
                    all_paths_grouped[b].append(collected_paths[path_counter])
                    path_counter += 1

        return all_paths_grouped, all_path_features

    def forward(self, node_features, adj_matrix, accident_mask=None):
        """
        Discover potential paths
        
        Args:
            node_features: (batch_size, seq_len, num_nodes, feat_dim)
            adj_matrix: (batch_size, num_nodes, num_nodes)
            accident_mask: Optional (batch_size, num_nodes) indicating which nodes have accidents
            
        Returns:
            paths: Discovered paths
            path_weights: Path importance weights
            path_features: Path feature representations
        """
        # Use random walks for structured path sampling
        paths, path_features = self._structural_path_sampling(
            node_features, adj_matrix, accident_mask
        )

        # Path scoring
        path_scores = self.path_scorer(path_features)
        
        # Normalize weights
        path_weights = F.softmax(path_scores, dim=1)

        return paths, path_weights, path_features

    def _extract_path_features_batched(self, paths, node_features, batch_indices):
        """Extract path features in batch"""
        N, path_len = paths.shape
        batch_size, seq_len, num_nodes, feat_dim = node_features.shape
        hidden_dim = self.path_encoder.input_size
        device = paths.device

        # Get node features for each path
        batch_node_features = node_features[batch_indices]
        
        # Create indices for gather
        path_indices = paths.unsqueeze(1).unsqueeze(-1).expand(-1, seq_len, -1, feat_dim)
        
        # Use gather to extract features
        gathered_features = torch.gather(
            batch_node_features, 2, path_indices.to(device)
        )
        
        # Use features from the last time step
        last_features = gathered_features[:, -1, :, :]
        
        # Adjust feature dimensions to match GRU input
        if feat_dim != hidden_dim:
            if feat_dim != hidden_dim:
                raise ValueError(
                    f"Feature dimension ({feat_dim}) doesn't match GRU input size ({hidden_dim}). Add an adapter layer."
                )
            path_features_gru_input = last_features
        else:
            path_features_gru_input = last_features
            
        # Use GRU to encode path features
        if not self.training:
            with torch.no_grad():
                encoded_features, _ = self.path_encoder(path_features_gru_input)
        else:
            encoded_features, _ = self.path_encoder(path_features_gru_input)
            
        # Take the hidden state of the last time step
        encoded_paths = encoded_features[:, -1, :]
        
        return encoded_paths


class PathInfluenceEvaluator(nn.Module):
    """Evaluate path influence intensity and propagation patterns"""

    def __init__(self, feat_dim, hidden_dim=64):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # Path influence encoder
        self.path_encoder = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Gated influence network
        self.influence_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Path-specific attention
        self.path_attention = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=4, batch_first=True
        )

    def forward(self, node_features, paths, path_features, path_weights):
        """Evaluate path influence intensity"""
        batch_size, seq_len, num_nodes, feat_dim = node_features.shape
        num_paths = path_features.shape[1]

        # Last time step features
        current_features = node_features[:, -1]
        
        # Pre-allocate result tensor
        all_influences = torch.zeros(
            batch_size, num_nodes, feat_dim, device=node_features.device
        )

        with torch.set_grad_enabled(self.training):
            for b in range(batch_size):
                path_influences_sum = torch.zeros(
                    num_nodes, feat_dim, device=node_features.device
                )

                batch_paths = paths[b]
                batch_weights = path_weights[b]
                batch_current_features = current_features[b]

                for p in range(num_paths):
                    path = batch_paths[p]
                    path_weight = batch_weights[p]

                    # Get path node features
                    path_node_features = batch_current_features[path]
                    
                    # Attention calculation
                    query = batch_current_features.unsqueeze(1)
                    key = path_node_features.unsqueeze(0).expand(num_nodes, -1, -1)
                    
                    attended_path, _ = self.path_attention(query, key, key)
                    attended_path = attended_path.squeeze(1)
                    
                    # Calculate influence gate values
                    source_features = batch_current_features[path[0]].expand(num_nodes, -1)
                    influence_input = torch.cat([batch_current_features, source_features], dim=-1)
                    
                    influence_encoded = self.path_encoder(influence_input)
                    influence_gate = self.influence_gate(influence_encoded)
                    
                    # Apply gated influence and accumulate to result
                    path_influences_sum.add_(path_weight * influence_gate * attended_path)
                    
                # Store results for current batch
                all_influences[b] = path_influences_sum

        return all_influences


class DynamicPropagation(nn.Module):
    """Complete dynamic path propagation module"""

    def __init__(
        self,
        in_steps,
        c_in,
        node_num,
        supports=[],
        max_paths=8,
        max_path_length=5,
        hidden_dim=64,
        dropout=0.0,
    ):
        super().__init__()
        self.in_steps = in_steps
        self.c_in = c_in
        self.node_num = node_num
        self.supports = supports
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Feature projection
        self.feat_proj = nn.Linear(c_in, hidden_dim)

        # Path discovery module
        self.path_discovery = PathDiscovery(
            node_num=node_num,
            hidden_dim=hidden_dim,
            max_paths=max_paths,
            max_path_length=max_path_length,
        )

        # Influence assessment module
        self.influence_assessment = PathInfluenceEvaluator(
            feat_dim=hidden_dim, hidden_dim=hidden_dim
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c_in),
        )

        self.proj = nn.Conv2d(1, in_steps, kernel_size=1)

    def forward(self, x, accident_mask=None):
        """
        Forward propagation
        
        Args:
            x: (batch_size, seq_len, num_nodes, feat_dim) or (batch_size, num_nodes, feat_dim)
            accident_mask: Optional (batch_size, num_nodes) accident mask
        """
        # Record original shape and adjust input
        is_3d_input = len(x.shape) == 3
        if is_3d_input:
            x = x.unsqueeze(1)  # Add time dimension

        batch_size, seq_len, num_nodes, feat_dim = x.shape

        # Project features
        x_flat = x.reshape(-1, feat_dim)
        x_proj_flat = self.feat_proj(x_flat)
        x_proj = x_proj_flat.reshape(batch_size, seq_len, num_nodes, self.hidden_dim)

        # Get adjacency matrix and expand to batch size
        adj = self.supports[0].to(x.device)
        adj_batch = adj.unsqueeze(0).expand(batch_size, -1, -1)

        # Extract features from the last time step
        last_features = x_proj[:, -1]

        # 1. Discover paths
        paths, path_weights, path_features = self.path_discovery(
            x_proj, adj_batch, accident_mask
        )

        # 2. Evaluate path influence
        influence_features = self.influence_assessment(
            x_proj, paths, path_features, path_weights
        )

        # 3. Integrate features and output projection
        combined_features = torch.cat([last_features, influence_features], dim=-1)
        output = self.output_proj(combined_features.reshape(-1, self.hidden_dim * 2))
        output = output.reshape(batch_size, num_nodes, self.c_in)

        # Apply dropout
        if self.training and self.dropout > 0:
            output = F.dropout(output, self.dropout)
            
        output = self.proj(output.unsqueeze(1))
        
        return output


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`)"""

    def __init__(self, model_dim, num_heads=8, qkv_bias=False, fast=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.fast = fast

    def forward(self, x):
        # Calculate QKV projections at once
        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)

        # Get batch size and sequence length
        batch_size, steps, length = query.shape[0], query.shape[1], query.shape[2]
        head_dim = self.head_dim
        num_heads = self.num_heads

        # Reshape to [batch, length, heads, dim] format
        q = query.reshape(batch_size, steps, length, num_heads, head_dim)
        k = key.reshape(batch_size, steps, length, num_heads, head_dim)
        v = value.reshape(batch_size, steps, length, num_heads, head_dim)

        try:
            # Try to use PyTorch's optimized attention implementation
            context = F.scaled_dot_product_attention(q, k, v)
            context = context.reshape(batch_size, steps, length, -1)
        except (RuntimeError, AttributeError):
            # If not supported, fall back to manual implementation
            qs = q.permute(0, 2, 1, 3)  # [batch, heads, length, dim]
            ks = k.permute(0, 2, 1, 3)
            vs = v.permute(0, 2, 1, 3)

            # Calculate attention scores
            attention_scores = torch.matmul(qs, ks.transpose(-1, -2)) / math.sqrt(head_dim)
            attention_weights = F.softmax(attention_scores, dim=-1)

            # Apply attention weights
            context_heads = torch.matmul(attention_weights, vs)
            # Reshape back to original format
            context = context_heads.permute(0, 2, 1, 3).reshape(batch_size, steps, length, -1)

        # Apply output projection
        out = self.out_proj(context)
        
        return out


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        c_dim,
        feed_forward_dim=2048,
        num_heads=8,
        dropout=0,
        mask=False,
        fast=False,
    ):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask, fast)
        self.ln1 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.ln2 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.feed_forward = Mlp(
            in_features=model_dim,
            hidden_features=feed_forward_dim,
            act_layer=nn.ReLU,
            drop=dropout,
        )
        self.dropout = DropPath(dropout)
        # Optimize GLN calculation - use a single linear layer
        self.GLN = nn.Sequential(nn.ReLU(), nn.Linear(c_dim, 6 * model_dim, bias=True))
        self.model_dim = model_dim

    def forward(self, x, c):
        # Batch calculate all modulation parameters
        modulation_params = self.GLN(c)
        
        # Efficient chunking - one-time operation
        params = modulation_params.chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa = params[0], params[1], params[2]
        shift_mlp, scale_mlp, gate_mlp = params[3], params[4], params[5]
        
        # First residual block - self-attention
        x_norm1 = self.ln1(x)
        x_mod1 = x_norm1 * (1 + scale_msa) + shift_msa
        attn_out = self.attn(x_mod1)
        
        # Apply gating and dropout
        drop_prob = self.dropout.drop_prob
        if drop_prob > 0 and self.training:
            x = x + self.dropout(gate_msa * attn_out)
        else:
            x = x + (gate_msa * attn_out)
            
        # Second residual block - feed-forward network
        x_norm2 = self.ln2(x)
        x_mod2 = x_norm2 * (1 + scale_mlp) + shift_mlp
        ff_out = self.feed_forward(x_mod2)
        
        # Apply gating and dropout
        if drop_prob > 0 and self.training:
            x = x + self.dropout(gate_mlp * ff_out)
        else:
            x = x + (gate_mlp * ff_out)
            
        return x


class ConFormer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=0,
        dow_embedding_dim=0,
        node_embedding_dim=0,
        acc_embedding_dim=0,
        reg_embedding_dim=0,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        adp_dropout=0,
        dow_dropout=0,
        tod_dropout=0,
        supports=1,
        fast=False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.acc_embedding_dim = acc_embedding_dim
        self.reg_embedding_dim = reg_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + node_embedding_dim
            + acc_embedding_dim
            + reg_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.adp_dropout = adp_dropout
        self.dow_dropout = dow_dropout
        self.tod_dropout = tod_dropout
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        self.graph_propagate = DynamicPropagation(
            in_steps,
            self.model_dim,
            self.num_nodes,
            supports,
            max_paths=5,
            max_path_length=5,
            dropout=dropout,
        )

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if node_embedding_dim > 0:
            self.node_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, node_embedding_dim))
            )
        if acc_embedding_dim > 0:
            self.acc_embedding = nn.Embedding(2, acc_embedding_dim)
        if reg_embedding_dim > 0:
            self.reg_embedding = nn.Embedding(2, reg_embedding_dim)

        self.output_proj = nn.Linear(self.model_dim, out_steps * output_dim)

        self.attn_layers = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    self.model_dim,
                    feed_forward_dim,
                    num_heads,
                    dropout,
                    fast=fast,
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder = nn.ModuleList(
            [
                Mlp(
                    in_features=self.model_dim,
                    hidden_features=int(self.model_dim * 2),
                    act_layer=nn.ReLU,
                    drop=dropout,
                )
                for _ in range(3)
            ]
        )
        self.encoder_proj = nn.Linear(
            in_steps * self.model_dim,
            self.model_dim,
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim)
        batch_size = x.shape[0]
        emb = self.input_proj(x[..., : self.input_dim])
        features = [emb]
        
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding((x[:, :, :, 1] * self.steps_per_day).long())
            tod_emb = F.dropout(tod_emb, self.tod_dropout, training=self.training)
            features.append(tod_emb)
            
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(x[:, :, :, 2].long())
            dow_emb = F.dropout(dow_emb, self.dow_dropout, training=self.training)
            features.append(dow_emb)
            
        if self.node_embedding_dim > 0:
            node_emb = self.node_embedding.expand(size=(batch_size, *self.node_embedding.shape))
            node_emb = F.dropout(node_emb, self.adp_dropout, training=self.training)
            features.append(node_emb)
            
        if self.acc_embedding_dim > 0:
            acc_emb = self.acc_embedding((x[..., 3] > 0).long())
            features.append(acc_emb)
            
        if self.reg_embedding_dim > 0:
            reg_emb = self.reg_embedding((x[..., 4] > 0).long())
            features.append(reg_emb)

        x = torch.concat(features, -1)
        c = self.graph_propagate(x)
        
        for attn in self.attn_layers:
            x = attn(x, c)
            
        x = self.encoder_proj(x.transpose(1, 2).flatten(-2))
        
        for layer in self.encoder:
            x = x + layer(x)
            
        out = self.output_proj(x).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        
        return out
