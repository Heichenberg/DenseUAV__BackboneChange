import torch
import torch.nn as nn

from .utils import weights_init_classifier, weights_init_kaiming


class TokenChannelMixerBlock(nn.Module):
    def __init__(self, channels, num_tokens=10):
        super().__init__()
        hidden_channel_dim = max(channels // 4, 1)
        self.norm1 = nn.LayerNorm(channels)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, num_tokens),
            nn.GELU(),
            nn.Linear(num_tokens, num_tokens),
        )
        self.norm2 = nn.LayerNorm(channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channel_dim),
            nn.GELU(),
            nn.Linear(hidden_channel_dim, channels),
        )

        self.token_mlp.apply(weights_init_kaiming)
        self.channel_mlp.apply(weights_init_kaiming)

    def forward(self, x):
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.norm2(x)
        y = self.channel_mlp(y)
        x = x + y
        return x


class GeoTokenV2CAGMixerBase(nn.Module):
    def __init__(self, opt, mixer_depth):
        super().__init__()
        in_channels = opt.in_planes
        hidden_gate_dim = max(in_channels // 4, 1)
        self.num_query_tokens = 8
        self.num_tokens = self.num_query_tokens + 2
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
        self.mixers = nn.Sequential(
            *[TokenChannelMixerBlock(in_channels, self.num_tokens) for _ in range(mixer_depth)]
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
        global_token = x.mean(dim=(2, 3)).unsqueeze(1)

        feat = x.flatten(2).transpose(1, 2)
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        q = self.q_proj(query)
        k = self.k_proj(feat)
        v = self.v_proj(feat)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * self.scale, dim=-1)
        query_tokens = torch.matmul(attn, v)

        detail_map = self.detail_dwconv(x)
        detail_token = detail_map.mean(dim=(2, 3)).unsqueeze(1)
        return torch.cat([global_token, query_tokens, detail_token], dim=1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                "GeoTokenV2CAGMixerHead expects [B, C, H, W] feature map, got {}".format(
                    tuple(x.shape)
                )
            )

        batch = x.shape[0]
        tokens = self._make_tokens(x)
        gate = torch.sigmoid(self.gate_mlp(tokens))
        gated_tokens = tokens * gate
        mixed_tokens = self.mixers(gated_tokens)

        embedding_raw = self.embedding(mixed_tokens.reshape(batch, -1))
        embedding = self.bnneck(embedding_raw)
        cls = self.classifier(embedding)
        return [cls, embedding]


class GeoTokenV2CAGMixer2Head(GeoTokenV2CAGMixerBase):
    def __init__(self, opt):
        super().__init__(opt, mixer_depth=2)


class GeoTokenV2CAGMixer4Head(GeoTokenV2CAGMixerBase):
    def __init__(self, opt):
        super().__init__(opt, mixer_depth=4)


class GeoTokenV2CAGMixer8Head(GeoTokenV2CAGMixerBase):
    def __init__(self, opt):
        super().__init__(opt, mixer_depth=8)
