import torch.nn as nn
import timm
from .RKNet import RKNet
from .vmamba_wrapper import VMambaBaseBackbone, VMambaSmallBackbone, VMambaTinyBackbone, VMambaTinyMSGEBlockBackbone
import torch
import os


def _find_project_root():
    path = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.isdir(os.path.join(path, "pretrained")) and os.path.isdir(os.path.join(path, "models")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        path = parent


PROJECT_ROOT = _find_project_root()
DEFAULT_PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained", "backbones")

LOCAL_BACKBONE_WEIGHTS = {
    "RKNet": os.path.join(DEFAULT_PRETRAINED_DIR, "torchvision", "resnet50-0676ba61.pth"),
    "DeitS-224": os.path.join(DEFAULT_PRETRAINED_DIR, "timm", "deit_small_distilled_patch16_224-649709d9.pth"),
    "SwinB-224": os.path.join(DEFAULT_PRETRAINED_DIR, "timm", "swin_base_patch4_window7_224_22kto1k.pth"),
    "EfficientNet-B2": os.path.join(DEFAULT_PRETRAINED_DIR, "timm", "efficientnet_b2_ra-bcdf34b7.pth"),
}

def make_backbone(opt):
    backbone_model = Backbone(opt)
    return backbone_model


class Backbone(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.img_size = (opt.h,opt.w)
        self.backbone_name = opt.backbone
        self.backbone,self.output_channel = self.init_backbone(opt.backbone)
        

    def _resolve_pretrained_path(self, backbone):
        explicit_weight = getattr(self.opt, "backbone_weight", "")
        if explicit_weight:
            return explicit_weight
        return LOCAL_BACKBONE_WEIGHTS.get(backbone, "")

    def _load_timm_checkpoint(self, model, checkpoint_path, backbone):
        if not checkpoint_path:
            return model
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                "{} pretrained checkpoint not found: {}. Run scripts/download_backbone_weights.py first.".format(
                    backbone, checkpoint_path
                )
            )
        from timm.models.helpers import load_checkpoint
        load_checkpoint(model, checkpoint_path, strict=False)
        print("Load {} pretrained checkpoint from: {}".format(backbone, checkpoint_path))
        return model

    def _create_timm_backbone(self, model_name, output_channel, backbone, **kwargs):
        checkpoint_path = self._resolve_pretrained_path(backbone)
        backbone_model = timm.create_model(model_name, pretrained=False, **kwargs)
        backbone_model = self._load_timm_checkpoint(backbone_model, checkpoint_path, backbone)
        return backbone_model, output_channel

    def init_backbone(self, backbone):
        if backbone=="resnet50":
            backbone_model = timm.create_model('resnet50', pretrained=True)
            output_channel = 2048
        elif backbone=="RKNet":
            backbone_model = RKNet(pretrained=self._resolve_pretrained_path(backbone))
            output_channel = 2048
        elif backbone=="senet":
            backbone_model = timm.create_model('legacy_seresnet50', pretrained=True)
            output_channel = 2048
        elif backbone=="ViTS-224":
            backbone_model = timm.create_model("vit_small_patch16_224", pretrained=True, img_size=self.img_size)
            output_channel = 384
        elif backbone=="ViTS-384":
            backbone_model = timm.create_model("vit_small_patch16_384", pretrained=True)
            output_channel = 384
        elif backbone=="DeitS-224":
            backbone_model, output_channel = self._create_timm_backbone(
                "deit_small_distilled_patch16_224", 384, backbone
            )
        elif backbone=="DeitB-224":
            backbone_model = timm.create_model("deit_base_distilled_patch16_224", pretrained=True)
            output_channel = 384
        elif backbone=="Pvtv2b2":
            backbone_model = timm.create_model("pvt_v2_b2", pretrained=True)
            output_channel = 512
        elif backbone=="ViTB-224":
            backbone_model = timm.create_model("vit_base_patch16_224", pretrained=True)
            output_channel = 768
        elif backbone=="SwinB-224":
            backbone_model, output_channel = self._create_timm_backbone(
                "swin_base_patch4_window7_224", 1024, backbone
            )
        elif backbone=="Swinv2S-256":
            backbone_model = timm.create_model("swinv2_small_window8_256", pretrained=True)
            output_channel = 768
        elif backbone=="Swinv2T-256":
            backbone_model = timm.create_model("swinv2_tiny_window16_256", pretrained=True)
            output_channel = 768
        elif backbone=="Convnext-T":
            backbone_model = timm.create_model("convnext_tiny", pretrained=True)
            output_channel = 768
        elif backbone=="EfficientNet-B2":
            backbone_model, output_channel = self._create_timm_backbone(
                "efficientnet_b2", 1408, backbone
            )
        elif backbone=="EfficientNet-B3":
            backbone_model = timm.create_model("efficientnet_b3", pretrained=True)
            output_channel = 1536
        elif backbone=="EfficientNet-B5":
            backbone_model = timm.create_model("tf_efficientnet_b5", pretrained=True)
            output_channel = 2048
        elif backbone=="EfficientNet-B6":
            backbone_model = timm.create_model("tf_efficientnet_b6", pretrained=True)
            output_channel = 2304
        elif backbone=="vgg16":
            backbone_model = timm.create_model("vgg16", pretrained=True)
            output_channel = 512
        elif backbone=="cvt13":
            from .cvt import get_cvt_models
            backbone_model, channels = get_cvt_models(model_size="cvt13")
            output_channel = channels[-1]
            checkpoint_weight = "/home/dmmm/VscodeProject/FPI/pretrain_model/CvT-13-384x384-IN-22k.pth"
            backbone_model = self.load_checkpoints(checkpoint_weight, backbone_model)
        elif backbone=="VMamba-Tiny":
            backbone_model = VMambaTinyBackbone(pretrained=getattr(self.opt, "backbone_weight", ""))
            output_channel = backbone_model.output_channel
        elif backbone in ("VMamba-MSGE", "Vmamba-MSGE", "VMamba-Tiny-MSGEBlock"):
            backbone_model = VMambaTinyMSGEBlockBackbone(pretrained=getattr(self.opt, "backbone_weight", ""))
            output_channel = backbone_model.output_channel
        elif backbone=="VMamba-Tiny-Vector":
            backbone_model = VMambaTinyBackbone(pretrained="", output_mode="vector")
            output_channel = backbone_model.output_channel
        elif backbone=="VMamba-Small":
            backbone_model = VMambaSmallBackbone(pretrained="")
            output_channel = backbone_model.output_channel
        elif backbone=="VMamba-Small-Vector":
            backbone_model = VMambaSmallBackbone(pretrained="", output_mode="vector")
            output_channel = backbone_model.output_channel
        elif backbone=="VMamba-Base":
            backbone_model = VMambaBaseBackbone(pretrained="")
            output_channel = backbone_model.output_channel
        elif backbone=="VMamba-Base-Vector":
            backbone_model = VMambaBaseBackbone(pretrained="", output_mode="vector")
            output_channel = backbone_model.output_channel
        else:
            raise NameError("{} not in the backbone list!!!".format(backbone))
        return backbone_model,output_channel
    
    def load_checkpoints(self, checkpoint_path, model):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        filter_ckpt = {k: v for k, v in ckpt.items() if "pos_embed" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(filter_ckpt, strict=False)
        print("Load pretrained backbone checkpoint from:", checkpoint_path)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        return model

    def _forward_deit_distilled_tokens(self, image):
        x = self.backbone.patch_embed(image)
        batch_size = x.shape[0]
        cls_token = self.backbone.cls_token.expand(batch_size, -1, -1)
        dist_token = self.backbone.dist_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, dist_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        for block in self.backbone.blocks:
            x = block(x)
        x = self.backbone.norm(x)
        return x

    def forward(self, image):
        if self.backbone_name == "DeitS-224":
            return self._forward_deit_distilled_tokens(image)
        features = self.backbone.forward_features(image)
        if isinstance(features, (tuple, list)):
            if all(torch.is_tensor(feature) for feature in features):
                if all(feature.ndim == 2 for feature in features):
                    features = torch.stack(features, dim=1)
                else:
                    features = features[0]
            else:
                features = features[0]
        return features
