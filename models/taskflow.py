import torch.nn as nn
from .Backbone.backbone import make_backbone
from .Head.head import make_head
import os
import torch


def normalize_model_alias(opt):
    alias_map = {
        "VMamba-Tiny-_GeoTokenHeadV1": ("VMamba-Tiny", "GeoTokenHeadV1", ("global", "center", "context", "structure"), 7, 384, False, 0.1),
        "VMamba-Tiny-_GeoTokenHeadV1_G": ("VMamba-Tiny", "GeoTokenHeadV1", ("global",), 7, 384, False, 0.1),
        "VMamba-Tiny-_GeoTokenHeadV1_GC": ("VMamba-Tiny", "GeoTokenHeadV1", ("global", "center"), 7, 384, False, 0.1),
        "VMamba-Tiny-_GeoTokenHeadV1_GR": ("VMamba-Tiny", "GeoTokenHeadV1", ("global", "context"), 7, 384, False, 0.1),
        "VMamba-Tiny-_GeoTokenHeadV1_GS": ("VMamba-Tiny", "GeoTokenHeadV1", ("global", "structure"), 7, 384, False, 0.1),
        "VMamba-Tiny-_GeoTokenHeadV1_GCR": ("VMamba-Tiny", "GeoTokenHeadV1", ("global", "center", "context"), 7, 384, False, 0.1),
        "VMamba-Tiny-_GeoTokenHeadV1_GCRS": ("VMamba-Tiny", "GeoTokenHeadV1", ("global", "center", "context", "structure"), 7, 384, False, 0.1),
        "VMamba-Tiny-_GeoTokenHeadV1_GCR5_D384": ("VMamba-Tiny", "GeoTokenHeadV1", ("global", "center", "context"), 5, 384, False, 0.1),
        "VMamba-Tiny-_GeoTokenHeadV1_GCR5_D192": ("VMamba-Tiny", "GeoTokenHeadV1", ("global", "center", "context"), 5, 192, False, 0.1),
        "VMamba-Tiny-_GeoTokenHeadV1_GCR5_D192_GATE": ("VMamba-Tiny", "GeoTokenHeadV1", ("global", "center", "context"), 5, 192, True, 0.1),
    }
    alias = getattr(opt, "backbone", "")
    if alias in alias_map:
        real_backbone, real_head, active_tokens, context_size, context_dim, context_gate, context_gate_init = alias_map[alias]
        opt.backbone = real_backbone
        opt.head = real_head
        opt.geo_active_tokens = active_tokens
        opt.geo_context_size = context_size
        opt.geo_context_dim = context_dim
        opt.geo_context_gate = context_gate
        opt.geo_context_gate_init = context_gate_init
    return opt


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        opt = normalize_model_alias(opt)
        self.backbone = make_backbone(opt)
        opt.in_planes = self.backbone.output_channel
        self.head = make_head(opt)
        self.opt = opt

    def forward(self, drone_image, satellite_image):
        if drone_image is None:
            drone_res = None
        else:
            drone_features = self.backbone(drone_image)
            drone_res = self.head(drone_features)
        if satellite_image is None:
            satellite_res = None
        else:
            satellite_features = self.backbone(satellite_image)
            satellite_res = self.head(satellite_features)
        
        return drone_res,satellite_res
    
    def load_params(self, load_from):
        pretran_model = torch.load(load_from)
        model2_dict = self.state_dict()
        state_dict = {k: v for k, v in pretran_model.items() if k in model2_dict.keys() and v.size() == model2_dict[k].size()}
        model2_dict.update(state_dict)
        self.load_state_dict(model2_dict)


def make_model(opt):
    model = Model(opt)
    if os.path.exists(opt.load_from):
        model.load_params(opt.load_from)
    return model
