import torch.nn as nn
import torch
import numpy as np
from timm.models.vision_transformer import Mlp, DropPath
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = model_dim // num_heads

        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=False)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x, dim=2):
        x = x.transpose(dim, 2)
        query, key, value = self.qkv(x).chunk(3, -1)
        qs = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=-3)
        ks = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=-3)
        vs = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=-3)
        x = (
            torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        x = self.out_proj(x)
        x = x.transpose(dim, 2)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        c_dim,
        mlp_ratio=2,
        num_heads=8,
        dropout=0,
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads)
        self.ln1 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.ln2 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.ln3 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.feed_forward = Mlp(
            in_features=model_dim,
            hidden_features=int(model_dim * mlp_ratio),
            act_layer=nn.ReLU,
            drop=dropout,
        )
        self.dropout = DropPath(dropout)
        self.GLN = nn.Sequential(nn.ReLU(), nn.Linear(c_dim, 9 * model_dim, bias=True))
        nn.init.constant_(self.GLN[-1].weight, 0)
        nn.init.constant_(self.GLN[-1].bias, 0)

    def forward(self, x, c):
        (
            shift_msa_s,
            scale_msa_s,
            gate_msa_s,
            shift_msa_t,
            scale_msa_t,
            gate_msa_t,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.GLN(c).chunk(9, dim=-1)
        x = x + self.dropout(
            gate_msa_s
            * self.attn(modulate(self.ln1(x), shift_msa_s, scale_msa_s), dim=2)
        )
        x = x + self.dropout(
            gate_msa_t
            * self.attn(modulate(self.ln2(x), shift_msa_t, scale_msa_t), dim=1)
        )
        x = x + self.dropout(
            gate_mlp * self.feed_forward(modulate(self.ln3(x), shift_mlp, scale_mlp))
        )
        return x


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("nvc,vw->nwc", (x, A.to(x)))
        return x.contiguous()


class GCN(nn.Module):
    def __init__(self, c_in, node_num, supports=[], adp=1, order=2, dropout=0.0):
        super(GCN, self).__init__()
        self.nconv = nconv()
        self.support_len = len(supports)
        self.supports = supports
        self.adp = adp
        self.dropout = dropout
        if self.support_len > 0:
            c_out = c_in
            c_in = (order * (self.support_len + adp) + 1) * c_in
            self.mlp = nn.Sequential(nn.Linear(c_in, c_out, bias=True), nn.ReLU())
            self.order = order
        if self.adp:
            self.nodevec1 = nn.Parameter(torch.randn(node_num, 10), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(10, node_num), requires_grad=True)

    def forward(self, x):
        supports = self.supports
        if self.support_len:
            if self.adp:
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                supports = self.supports + [adp]
            out = [x]
            for a in supports:
                x1 = self.nconv(x, a)
                out.append(x1)
                for k in range(2, self.order + 1):
                    x2 = self.nconv(x1, a)
                    out.append(x2)
                    x1 = x2
            h = self.mlp(torch.cat(out, dim=-1))
            h = F.dropout(h, self.dropout, training=self.training)
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
        tod_embedding_dim=12,
        dow_embedding_dim=12,
        adaptive_embedding_dim=12,
        acc_embedding_dim=0,
        reg_embedding_dim=0,
        num_heads=4,
        supports=None,
        num_layers=3,
        dropout=0.1,
        adp_dropout=0.1,
        dow_dropout=0.0,
        tod_dropout=0.0,
        mlp_ratio=2,
        use_mixed_proj=True,
        kernel_size=[1],
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
        self.acc_embedding_dim = acc_embedding_dim
        self.reg_embedding_dim = reg_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + adaptive_embedding_dim
            + acc_embedding_dim
            + reg_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.graph_propagate = GCN(
            self.model_dim - input_embedding_dim,
            self.num_nodes,
            supports,
            adp=True,
            dropout=dropout,
        )

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        if acc_embedding_dim > 0:
            self.acc_embedding = nn.Embedding(2, acc_embedding_dim)
        if reg_embedding_dim > 0:
            self.reg_embedding = nn.Embedding(2, reg_embedding_dim)

        self.adp_dropout = nn.Dropout(adp_dropout)

        self.attn_layers_st = SelfAttentionLayer(
            self.model_dim,
            self.model_dim - input_embedding_dim,
            mlp_ratio,
            num_heads,
            dropout,
        )

        self.encoder_proj = nn.Linear(
            (in_steps - sum(k - 1 for k in kernel_size)) * self.model_dim,
            self.model_dim,
        )
        self.kernel_size = kernel_size[0]

        self.encoder = nn.ModuleList(
            [
                Mlp(
                    in_features=self.model_dim,
                    hidden_features=int(self.model_dim * mlp_ratio),
                    act_layer=nn.ReLU,
                    drop=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(self.model_dim, out_steps * output_dim)
        # self.temporal_proj = TCNLayer(self.model_dim, self.model_dim, max_seq_length=in_steps)
        self.temporal_proj = nn.Conv2d(
            self.model_dim, self.model_dim, (1, kernel_size[0]), 1, 0
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
        c = torch.tensor([]).to(x)
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            c = torch.concat([c, tod_emb], -1)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            c = torch.concat([c, dow_emb], -1)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            c = torch.concat([c, self.adp_dropout(adp_emb)], -1)
        if self.acc_embedding_dim > 0:
            acc_emb = self.acc_embedding((x[..., 3] > 0).long())
            c = torch.concat([c, acc_emb], -1)
        if self.reg_embedding_dim > 0:
            reg_emb = self.reg_embedding((x[..., 4] > 0).long())
            c = torch.concat([c, reg_emb], -1)

        x = torch.cat([x] + [c], dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        x = self.temporal_proj(x.transpose(1, 3)).transpose(1, 3)
        x = self.attn_layers_st(x, c)
        x = self.encoder_proj(x.transpose(1, 2).flatten(-2))
        for layer in self.encoder:
            x = x + layer(x) 
        out = self.output_proj(x).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        return out
