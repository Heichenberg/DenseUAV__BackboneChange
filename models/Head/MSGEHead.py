import torch
import torch.nn as nn

from .utils import weights_init_classifier, weights_init_kaiming


class MSGE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_dim = max(channels // reduction, 1)
        self.dwconv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dwconv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.dwconv7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.fusion = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

        self._init_laplacian_depthwise(self.dwconv3, 3)
        self._init_laplacian_depthwise(self.dwconv5, 5)
        self._init_laplacian_depthwise(self.dwconv7, 7)
        self.fusion.apply(weights_init_kaiming)
        self.se.apply(weights_init_kaiming)

    def _init_laplacian_depthwise(self, conv, kernel_size):
        weight = conv.weight.data
        weight.zero_()
        kernel = weight.new_full((kernel_size, kernel_size), -1.0 / (kernel_size * kernel_size - 1))
        center = kernel_size // 2
        kernel[center, center] = 1.0
        weight[:, 0, :, :] = kernel

    def get_alpha(self):
        return float(self.alpha.detach().cpu().item())

    def forward(self, x):
        g3 = self.dwconv3(x)
        g5 = self.dwconv5(x)
        g7 = self.dwconv7(x)
        g_cat = torch.cat([g3, g5, g7], dim=1)
        g = self.fusion(g_cat)
        w = self.se(g)
        g_enhanced = g * w
        return x + self.alpha * g_enhanced


class MSGEHead(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_channels = opt.in_planes
        self.embedding_dim = opt.num_bottleneck

        self.msge = MSGE(in_channels)
        self.embedding = nn.Linear(in_channels, self.embedding_dim)
        self.bnneck = nn.BatchNorm1d(self.embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.embedding_dim, opt.nclasses)

        self.embedding.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def get_msge_alpha(self):
        return self.msge.get_alpha()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("MSGEHead expects [B, C, H, W] feature map, got {}".format(tuple(x.shape)))

        x_enhanced = self.msge(x)
        feat = x_enhanced.mean(dim=(2, 3))
        embedding_raw = self.embedding(feat)
        embedding = self.bnneck(embedding_raw)
        cls = self.classifier(embedding)
        return [cls, embedding]
