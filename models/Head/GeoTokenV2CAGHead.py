import torch
import torch.nn as nn

from .utils import weights_init_classifier, weights_init_kaiming


class GeoTokenV2CAGHead(nn.Module):
    """
    GeoTokenV2 with content-adaptive token-level gate.

    External contract matches GeoTokenV2Head:
    input  [B, C, H, W]
    output [cls, embedding]
    """

    def __init__(self, opt, use_global=True, use_detail=True):
        super().__init__()
        in_channels = opt.in_planes
        if not use_global and not use_detail:
            raise ValueError("GeoTokenV2CAGHead requires at least one of global/detail tokens")
        self.use_global = use_global
        self.use_detail = use_detail
        hidden_gate_dim = max(in_channels // 4, 1)
        self.num_query_tokens = 8
        self.num_tokens = self.num_query_tokens + int(use_global) + int(use_detail)
        self.embedding_dim = opt.num_bottleneck
        self.scale = in_channels ** -0.5

        self.query = nn.Parameter(torch.randn(self.num_query_tokens, in_channels))
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.detail_dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_gate_dim),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 1),
        )
        self.embedding = nn.Linear(self.num_tokens * in_channels, self.embedding_dim)
        self.bnneck = nn.BatchNorm1d(self.embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.embedding_dim, opt.nclasses)

        nn.init.normal_(self.query, std=0.02)
        self.q_proj.apply(weights_init_kaiming)
        self.k_proj.apply(weights_init_kaiming)
        self.v_proj.apply(weights_init_kaiming)
        self.detail_dwconv.apply(weights_init_kaiming)
        self.gate_mlp.apply(weights_init_kaiming)
        self.embedding.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def _make_tokens(self, x):
        batch = x.shape[0]
        tokens = []
        if self.use_global:
            tokens.append(x.mean(dim=(2, 3)).unsqueeze(1))

        feat = x.flatten(2).transpose(1, 2)
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        q = self.q_proj(query)
        k = self.k_proj(feat)
        v = self.v_proj(feat)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * self.scale, dim=-1)
        tokens.append(torch.matmul(attn, v))

        if self.use_detail:
            detail_map = self.detail_dwconv(x)
            tokens.append(detail_map.mean(dim=(2, 3)).unsqueeze(1))

        return torch.cat(tokens, dim=1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                "GeoTokenV2CAGHead expects [B, C, H, W] feature map, got {}".format(
                    tuple(x.shape)
                )
            )

        batch = x.shape[0]
        tokens = self._make_tokens(x)
        gate = torch.sigmoid(self.gate_mlp(tokens))
        gated_tokens = tokens * gate

        embedding_raw = self.embedding(gated_tokens.reshape(batch, -1))
        embedding = self.bnneck(embedding_raw)
        cls = self.classifier(embedding)
        return [cls, embedding]


class GeoTokenV2CAGNoGlobalHead(GeoTokenV2CAGHead):
    def __init__(self, opt):
        super().__init__(opt, use_global=False, use_detail=True)


class GeoTokenV2CAGNoDetailHead(GeoTokenV2CAGHead):
    def __init__(self, opt):
        super().__init__(opt, use_global=True, use_detail=False)
