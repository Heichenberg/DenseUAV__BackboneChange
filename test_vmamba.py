import torch
from models.Backbone.vmamba_wrapper import VMambaTinyBackbone

# Test VMamba Tiny Backbone
model = VMambaTinyBackbone(pretrained="pretrained/backbones/vmamba/tiny/vssm1_tiny_0230s_ckpt_epoch_264.pth")
model.eval()

# Dummy input
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(x)
    print("Output shape:", output.shape)
    print("Success!")