import torch.nn as nn
import torch
from torchinfo import summary
from timm.models.layers import DropPath
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, qkv_bias=False, fast = False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = model_dim // num_heads

        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)

        self.out_proj = nn.Linear(model_dim, model_dim)

        self.fast = fast

    def forward(self, x):
        query, key, value = self.qkv(x).chunk(3, -1)
        if self.fast:
            qs = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=-2).flatten(start_dim=0, end_dim=1)
            ks = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=-2).flatten(start_dim=0, end_dim=1)
            vs = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=-2).flatten(start_dim=0, end_dim=1)

            qs = nn.functional.normalize(qs, dim=-1)
            ks = nn.functional.normalize(ks, dim=-1)
            N = qs.shape[1]
            batch_size = query.shape[0]
            length = query.shape[1]

            # numerator
            kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
            attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)  # [N, H, D]
            attention_num += N * vs

            # denominator
            all_ones = torch.ones([ks.shape[1]]).to(ks.device)
            ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones)
            attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)  # [N, H]

            # attentive aggregated results
            attention_normalizer = torch.unsqueeze(
                attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
            attention_normalizer += torch.ones_like(attention_normalizer) * N
            out = attention_num / attention_normalizer  # [N, H, D]
            out = torch.unflatten(out, 0, (batch_size, length)).flatten(start_dim=3)
            out = self.out_proj(out)
        else:
            qs = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=2)
            ks = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=2)
            vs = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=2)
            out = torch.nn.functional.scaled_dot_product_attention(qs, ks, vs).transpose(3, 2).flatten(start_dim=3) 
            out = self.out_proj(out)

        return out

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, support_len = 1, dropout=0, mask=False, mode = 1, fast = False
    ):
        super().__init__()
    
        self.attn = AttentionLayer(model_dim, num_heads, mask, fast )
        self.ln1 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        if mode == 2:
            self.ln2 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
            self.feed_forward = nn.Sequential(
                nn.Linear(model_dim, feed_forward_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feed_forward_dim, model_dim),
            )
        self.dropout = DropPath(dropout)
        self.GLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim * (1 + support_len), 6 * model_dim, bias=True)
        )

    def forward(self, x, c, dim=-2):
        x, c = x.transpose(dim, -2), c.transpose(dim, -2)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.GLN(c).chunk(6, dim=-1)
        x = x + self.dropout(gate_msa * self.attn(modulate(self.ln1(x), shift_msa, scale_msa)))
        if dim == 2:
           x = x + self.dropout(gate_mlp * self.feed_forward(modulate(self.ln2(x), shift_mlp, scale_mlp)))
        x = x.transpose(dim, -2)
        return x
    

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()


    def forward(self, x, A):
        x = torch.einsum('nlvc,vw->nlwc',(x, A))
        return x.contiguous()


class GCN(nn.Module):
    def __init__(self, c_in, c_out, node_num, support_len=1, order=1):
        super(GCN, self).__init__()
        self.nconv = nconv()
        self.support_len = support_len
        if support_len > 0:
            c_in = (order * support_len + 1) * c_in
            self.mlp = nn.Linear(c_in, c_out)
            self.order = order
            self.nodevec1 = nn.Parameter(torch.randn(node_num, 10), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(10, node_num), requires_grad=True)

    def forward(self, x):
        if self.support_len:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            out = [x]
            for a in [adp]:
                x1 = self.nconv(x, a)
                out.append(x1)
                for k in range(2, self.order + 1):
                    x2 = self.nconv(x1, a)
                    out.append(x2)
                    x1 = x2
            h = torch.cat(out, dim=-1) #self.mlp(torch.cat(out, dim=-1))
            # print(x.shape, h.shape)
        else:
            h = x
        return h
    
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
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        adaptive_embedding_dim=24,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        adp_dropout=0,
        support_len=1,
        use_mixed_proj=True,
        fast = False
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
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.dropout = adp_dropout

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.graph_propagate = GCN(self.model_dim, self.model_dim, self.num_nodes, support_len)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)


        self.attn_layers_st = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, support_len, dropout, mode = i % 2 + 1, fast = fast)
                for i in range(num_layers)
            ]
        )


    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            adp_emb = F.dropout(adp_emb, self.dropout, training=self.training)
            features.append(adp_emb)
        c = x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        c = self.graph_propagate(x)
        for i, attn in enumerate(self.attn_layers_st):
            x = attn(x, c, dim = i % 2 + 1)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out


if __name__ == "__main__":
    model = ConFormer(307, 12, 12, num_layers=6)
    # summary(model, [64, 12, 207, 3])
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    inputs = torch.rand(1, 12, 307, 96)
    print(flop_count_table(FlopCountAnalysis(model, inputs)))