import os
import sys
import urllib.request


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

WEIGHTS = {
    "RKNet": (
        "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        os.path.join(PROJECT_ROOT, "pretrained", "backbones", "torchvision", "resnet50-0676ba61.pth"),
    ),
    "DeitS-224": (
        "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
        os.path.join(PROJECT_ROOT, "pretrained", "backbones", "timm", "deit_small_distilled_patch16_224-649709d9.pth"),
    ),
    "SwinB-224": (
        "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth",
        os.path.join(PROJECT_ROOT, "pretrained", "backbones", "timm", "swin_base_patch4_window7_224_22kto1k.pth"),
    ),
    "EfficientNet-B2": (
        "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth",
        os.path.join(PROJECT_ROOT, "pretrained", "backbones", "timm", "efficientnet_b2_ra-bcdf34b7.pth"),
    ),
}


def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print("exists:", path)
        return
    tmp_path = path + ".tmp"
    print("download:", url)
    print("to:", path)
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, path)


def main():
    names = sys.argv[1:] or list(WEIGHTS.keys())
    unknown = [name for name in names if name not in WEIGHTS]
    if unknown:
        raise SystemExit("Unknown backbone(s): {}".format(", ".join(unknown)))
    for name in names:
        url, path = WEIGHTS[name]
        download(url, path)


if __name__ == "__main__":
    main()
