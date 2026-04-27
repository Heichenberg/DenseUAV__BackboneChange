import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weights_init_kaiming, weights_init_classifier


class MixerMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, dim, token_mlp_dim, channel_mlp_dim, drop_rate=0.0):
        super().__init__()
        self.norm_tokens = nn.LayerNorm(dim)
        self.token_mlp = MixerMlp(num_tokens, token_mlp_dim, drop_rate)
        self.norm_channels = nn.LayerNorm(dim)
        self.channel_mlp = MixerMlp(dim, channel_mlp_dim, drop_rate)

    def forward(self, x):
        y = self.norm_tokens(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.norm_channels(x)
        y = self.channel_mlp(y)
        x = x + y
        return x


class MixerHeadV1(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_planes = opt.in_planes
        self.num_tokens = getattr(opt, "mixer_num_tokens", 49)
        self.proj_dim = getattr(opt, "mixer_proj_dim", 384)
        self.token_out = getattr(opt, "mixer_token_out", 4)
        self.mixer_depth = getattr(opt, "mixer_depth", 2)
        self.drop_rate = getattr(opt, "mixer_drop_rate", opt.droprate)
        token_mlp_dim = getattr(opt, "mixer_token_mlp_dim", max(self.num_tokens * 2, 64))
        channel_mlp_dim = getattr(opt, "mixer_channel_mlp_dim", self.proj_dim * 2)
        embedding_dim = opt.num_bottleneck

        self.input_norm = nn.LayerNorm(in_planes)
        self.channel_proj = nn.Linear(in_planes, self.proj_dim)
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    num_tokens=self.num_tokens,
                    dim=self.proj_dim,
                    token_mlp_dim=token_mlp_dim,
                    channel_mlp_dim=channel_mlp_dim,
                    drop_rate=self.drop_rate,
                )
                for _ in range(self.mixer_depth)
            ]
        )
        self.output_norm = nn.LayerNorm(self.proj_dim)
        self.token_proj = nn.Linear(self.num_tokens, self.token_out)
        self.embedding = nn.Linear(self.token_out * self.proj_dim, embedding_dim)
        self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embedding_dim, opt.nclasses)

        self.channel_proj.apply(weights_init_kaiming)
        self.embedding.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError("MixerHeadV1 expects [B, C, H, W], got {}".format(tuple(features.shape)))

        x = features.flatten(2).transpose(1, 2)
        if x.shape[1] != self.num_tokens:
            raise ValueError("MixerHeadV1 expects {} tokens, got {}".format(self.num_tokens, x.shape[1]))

        x = self.input_norm(x)
        x = self.channel_proj(x)
        x = self.blocks(x)
        x = self.output_norm(x)

        x = x.transpose(1, 2)
        x = self.token_proj(x)
        x = x.transpose(1, 2).reshape(x.shape[0], -1)

        feature = self.embedding(x)
        feature = self.bnneck(feature)
        cls = self.classifier(feature)
        return [cls, feature]
