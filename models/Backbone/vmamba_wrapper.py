import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


def _repo_root():
    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "models").is_dir() and (candidate / "pretrained").is_dir():
            return candidate
    for candidate in candidates:
        if (candidate / "models").is_dir() and (candidate / "third_party").is_dir():
            return candidate
    return Path(__file__).resolve().parents[2]


def _resolve_path(path):
    if not path:
        return None
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return _repo_root() / path


def _ensure_fvcore_stub():
    try:
        import fvcore.nn  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    fvcore = types.ModuleType("fvcore")
    fvcore_nn = types.ModuleType("fvcore.nn")
    fvcore_nn.FlopCountAnalysis = object
    fvcore_nn.flop_count_str = lambda *args, **kwargs: ""
    fvcore_nn.flop_count = lambda *args, **kwargs: ({}, {})
    fvcore_nn.parameter_count = lambda *args, **kwargs: {"": 0}
    fvcore.nn = fvcore_nn
    sys.modules.setdefault("fvcore", fvcore)
    sys.modules.setdefault("fvcore.nn", fvcore_nn)


def _cross_scan_fwd(x, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        bsz, channels, height, width = x.shape
        if scans == 0:
            y = x.new_empty((bsz, 4, channels, height * width))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(bsz, 1, channels, height * width).repeat(1, 4, 1, 1)
        elif scans == 2:
            y = x.view(bsz, 1, channels, height * width).repeat(1, 2, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
        else:
            raise ValueError("Unsupported scan mode: {}".format(scans))
    else:
        bsz, height, width, channels = x.shape
        if scans == 0:
            y = x.new_empty((bsz, height * width, 4, channels))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = x.transpose(dim0=1, dim1=2).flatten(1, 2)
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])
        elif scans == 1:
            y = x.view(bsz, height * width, 1, channels).repeat(1, 1, 4, 1)
        elif scans == 2:
            y = x.view(bsz, height * width, 1, channels).repeat(1, 1, 2, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)
        else:
            raise ValueError("Unsupported scan mode: {}".format(scans))

    if in_channel_first and not out_channel_first:
        y = y.permute(0, 3, 1, 2).contiguous()
    elif not in_channel_first and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()
    return y


def _cross_merge_fwd(y, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        bsz, groups, channels, height, width = y.shape
        y = y.view(bsz, groups, channels, -1)
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(bsz, 2, channels, -1)
            y = y[:, 0] + y[:, 1].view(bsz, -1, width, height).transpose(dim0=2, dim1=3).contiguous().view(bsz, channels, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(bsz, 2, channels, -1)
            y = y.sum(1)
        else:
            raise ValueError("Unsupported scan mode: {}".format(scans))
    else:
        bsz, height, width, groups, channels = y.shape
        y = y.view(bsz, -1, groups, channels)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(bsz, -1, 2, channels)
            y = y[:, :, 0] + y[:, :, 1].view(bsz, width, height, -1).transpose(dim0=1, dim1=2).contiguous().view(bsz, -1, channels)
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(bsz, -1, 2, channels)
            y = y.sum(2)
        else:
            raise ValueError("Unsupported scan mode: {}".format(scans))

    if in_channel_first and not out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    elif not in_channel_first and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    return y


def _ensure_csm_triton_stub():
    try:
        import triton  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    csm_triton = types.ModuleType("csm_triton")
    csm_triton.WITH_TRITON = False
    csm_triton.cross_scan_fn = lambda x, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False: _cross_scan_fwd(
        x, in_channel_first, out_channel_first, scans
    )
    csm_triton.cross_merge_fn = lambda y, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False: _cross_merge_fwd(
        y, in_channel_first, out_channel_first, scans
    )
    sys.modules.setdefault("csm_triton", csm_triton)
    sys.modules.setdefault("third_party.vmamba.csm_triton", csm_triton)


def _import_vmamba_builders():
    _ensure_fvcore_stub()
    _ensure_csm_triton_stub()
    from third_party.vmamba.classification.models import csms6s
    from third_party.vmamba.classification.models import vmamba
    from third_party.vmamba.classification.models.vmamba import (
        vmamba_base_s2l15,
        vmamba_small_s2l15,
        vmamba_tiny_s1l8,
    )

    original_cross_scan_fn = vmamba.cross_scan_fn
    original_cross_merge_fn = vmamba.cross_merge_fn
    original_selective_scan_fn = csms6s.selective_scan_fn

    def cpu_safe_cross_scan_fn(x, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
        if x.is_cuda:
            return original_cross_scan_fn(x, in_channel_first, out_channel_first, one_by_one, scans, force_torch)
        return _cross_scan_fwd(x, in_channel_first, out_channel_first, scans)

    def cpu_safe_cross_merge_fn(y, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
        if y.is_cuda:
            return original_cross_merge_fn(y, in_channel_first, out_channel_first, one_by_one, scans, force_torch)
        return _cross_merge_fwd(y, in_channel_first, out_channel_first, scans)

    def cpu_safe_selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, oflex=True, backend=None):
        if not u.is_cuda:
            backend = "torch"
        return original_selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)

    vmamba.cross_scan_fn = cpu_safe_cross_scan_fn
    vmamba.cross_merge_fn = cpu_safe_cross_merge_fn
    vmamba.selective_scan_fn = cpu_safe_selective_scan_fn
    csms6s.selective_scan_fn = cpu_safe_selective_scan_fn
    return {
        "tiny": vmamba_tiny_s1l8,
        "small": vmamba_small_s2l15,
        "base": vmamba_base_s2l15,
    }


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
        self._init_conv(self.fusion)
        self.se.apply(self._init_conv)

    def _init_laplacian_depthwise(self, conv, kernel_size):
        weight = conv.weight.data
        weight.zero_()
        kernel = weight.new_full((kernel_size, kernel_size), -1.0 / (kernel_size * kernel_size - 1))
        center = kernel_size // 2
        kernel[center, center] = 1.0
        weight[:, 0, :, :] = kernel

    def _init_conv(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_alpha(self):
        return float(self.alpha.detach().cpu().item())

    def forward(self, x):
        g3 = self.dwconv3(x)
        g5 = self.dwconv5(x)
        g7 = self.dwconv7(x)
        g_cat = torch.cat([g3, g5, g7], dim=1)
        g_fused = self.fusion(g_cat)
        g_enhanced = g_fused * self.se(g_fused)
        return x + self.alpha * g_enhanced


def _make_msge_vss_block(base_vss_block):
    class MSGEVSSBlock(base_vss_block):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            hidden_dim = kwargs.get("hidden_dim", args[0] if args else 0)
            channel_first = kwargs.get("channel_first", False)
            if not channel_first:
                raise ValueError("MSGEVSSBlock currently requires channel_first=True")
            self.msge = MSGE(hidden_dim)

        def _forward(self, input):
            x = input
            if self.ssm_branch:
                if self.post_norm:
                    y = self.op(x)
                    y = self.msge(y)
                    x = x + self.drop_path(self.norm(y))
                else:
                    y = self.op(self.norm(x))
                    y = self.msge(y)
                    x = x + self.drop_path(y)
            if self.mlp_branch:
                if self.post_norm:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

    return MSGEVSSBlock


class VMambaBackbone(nn.Module):
    _BUILDERS = None

    def __init__(self, variant="tiny", pretrained="", output_mode="map", block_factory=None):
        super().__init__()
        if self._BUILDERS is None:
            type(self)._BUILDERS = _import_vmamba_builders()
        if variant not in self._BUILDERS:
            raise ValueError("Unsupported VMamba variant: {}".format(variant))
        if output_mode not in {"map", "vector"}:
            raise ValueError("Unsupported VMamba output mode: {}".format(output_mode))

        self.variant = variant
        self.output_mode = output_mode
        self.backbone = self._build_backbone(variant, block_factory)
        self.output_channel = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.output_norm = nn.LayerNorm(self.output_channel)

        if pretrained:
            self.load_pretrained(pretrained)

    def _build_backbone(self, variant, block_factory=None):
        if block_factory is None:
            return self._BUILDERS[variant](channel_first=True)

        from third_party.vmamba.classification.models import vmamba

        original_vss_block = vmamba.VSSBlock
        vmamba.VSSBlock = block_factory(original_vss_block)
        try:
            return self._BUILDERS[variant](channel_first=True)
        finally:
            vmamba.VSSBlock = original_vss_block

    def load_pretrained(self, checkpoint_path):
        checkpoint_path = _resolve_path(checkpoint_path)
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError("VMamba checkpoint not found: {}".format(checkpoint_path))

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier.head")}
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        print("Load pretrained VMamba checkpoint from:", checkpoint_path)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)

    def forward_features(self, x):
        x = self.backbone.patch_embed(x)
        if self.backbone.pos_embed is not None:
            pos_embed = self.backbone.pos_embed
            if not self.backbone.channel_first:
                pos_embed = pos_embed.permute(0, 2, 3, 1)
            x = x + pos_embed

        for layer in self.backbone.layers:
            x = layer(x)

        if not self.backbone.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()

        if self.output_mode == "vector":
            x = self.global_pool(x).reshape(x.shape[0], -1)
            x = self.output_norm(x)

        return x

    def forward(self, x):
        return self.forward_features(x)


class VMambaTinyBackbone(VMambaBackbone):
    def __init__(self, pretrained="", output_mode="map"):
        super().__init__(variant="tiny", pretrained=pretrained, output_mode=output_mode)


class VMambaTinyMSGEBlockBackbone(VMambaBackbone):
    def __init__(self, pretrained="", output_mode="map"):
        super().__init__(
            variant="tiny",
            pretrained=pretrained,
            output_mode=output_mode,
            block_factory=_make_msge_vss_block,
        )


class VMambaSmallBackbone(VMambaBackbone):
    def __init__(self, pretrained="", output_mode="map"):
        super().__init__(variant="small", pretrained=pretrained, output_mode=output_mode)


class VMambaBaseBackbone(VMambaBackbone):
    def __init__(self, pretrained="", output_mode="map"):
        super().__init__(variant="base", pretrained=pretrained, output_mode=output_mode)
