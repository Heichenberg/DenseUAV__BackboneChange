import torch.optim as optim
from torch.optim import lr_scheduler


def make_optimizer(model,opt):
    backbone_lr = opt.backbone_lr
    head_lr = opt.head_lr
    if getattr(opt, "backbone", "") != "VMamba-Tiny":
        backbone_lr = opt.lr
        head_lr = opt.lr
        opt.backbone_lr = opt.lr
        opt.head_lr = opt.lr

    backbone_params = []
    head_params = []
    seen_params = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen_params:
            raise ValueError("Duplicate parameter detected while building optimizer: {}".format(name))
        seen_params.add(param_id)
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    if len(backbone_params) == 0:
        raise ValueError("No trainable backbone parameters found for optimizer group 0")
    if len(head_params) == 0:
        raise ValueError("No trainable head/other parameters found for optimizer group 1")

    print("backbone lr = {}".format(backbone_lr))
    print("head lr = {}".format(head_lr))
    print("num backbone params = {}".format(sum(p.numel() for p in backbone_params)))
    print("num head params = {}".format(sum(p.numel() for p in head_params)))

    optimizer_ft = optim.SGD([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)


    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[70,110], gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.95)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=4, verbose=True,threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)

    return optimizer_ft,exp_lr_scheduler
