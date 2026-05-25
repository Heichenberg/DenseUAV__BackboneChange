import torch
import torch.nn as nn

from .utils import weights_init_classifier, weights_init_kaiming


class GeoTokenV2Head(nn.Module):
    """
    Stage 1 GeoTokenV2-A head.

    External contract matches GeoTokenHeadV1:
    input  [B, C, H, W]
    output [cls, embedding]
    """

    def __init__(self, opt):
        super().__init__()
        in_channels = opt.in_planes
        self.num_query_tokens = 8
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
        self.embedding = nn.Linear((self.num_query_tokens + 2) * in_channels, self.embedding_dim)
        self.bnneck = nn.BatchNorm1d(self.embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.embedding_dim, opt.nclasses)

        nn.init.normal_(self.query, std=0.02)
        self.q_proj.apply(weights_init_kaiming)
        self.k_proj.apply(weights_init_kaiming)
        self.v_proj.apply(weights_init_kaiming)
        self.detail_dwconv.apply(weights_init_kaiming)
        self.embedding.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                "GeoTokenV2Head expects [B, C, H, W] feature map, got {}".format(
                    tuple(x.shape)
                )
            )

        batch, channels, height, width = x.shape

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

        tokens = torch.cat([global_token, query_tokens, detail_token], dim=1)
        embedding_raw = self.embedding(tokens.reshape(batch, -1))
        embedding = self.bnneck(embedding_raw)
        cls = self.classifier(embedding)
        return [cls, embedding]
