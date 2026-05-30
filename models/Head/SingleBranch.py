import torch.nn as nn
from .utils import ClassBlock, Pooling
import torch.nn.functional as F
import torch

class SingleBranch(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.head_pool = opt.head_pool
        self.classifier = ClassBlock(
            opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)

    def forward(self, features):
        if features.ndim == 2:
            feature = features
            cls, feature = self.classifier(feature)
            return [cls, feature]
        global_feature = features[:, 0]
        local_feature = features[:, 1:]
        if self.head_pool == "global":
            feature = global_feature
        elif self.head_pool == "avg":
            local_feature = local_feature.transpose(1, 2)
            feature = torch.mean(local_feature, 2)
        elif self.head_pool == "max":
            local_feature = local_feature.transpose(1, 2)
            feature = torch.max(local_feature, 2)[0]
        elif self.head_pool == "avg+max":
            local_feature = local_feature.transpose(1, 2)
            avg_feature = torch.mean(local_feature, 2)
            max_feature = torch.max(local_feature, 2)[0]
            feature = avg_feature+max_feature
        else:
            raise TypeError("head_pool 不在支持的列表中！！！")

        cls, feature = self.classifier(feature)
        return [cls, feature]


class SingleBranchCNN(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassBlock(
            opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)

    def forward(self, features):
        global_feature = self.pool(features).reshape(features.shape[0], -1)
        cls, feature = self.classifier(global_feature)
        return [cls, feature]


class SingleBranchSwin(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(
            opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)

    def forward(self, features):
        if features.ndim == 2:
            global_feature = features
        elif features.ndim == 3:
            global_feature = self.pool(features.transpose(2, 1)).reshape(features.shape[0], -1)
        elif features.ndim == 4:
            if features.shape[1] == self.opt.in_planes:
                features = features.flatten(2).transpose(2, 1)
            elif features.shape[-1] == self.opt.in_planes:
                features = features.reshape(features.shape[0], -1, features.shape[-1])
            else:
                raise ValueError(
                    "SingleBranchSwin expected 4D features with channel dimension {}, got {}".format(
                        self.opt.in_planes, tuple(features.shape)
                    )
                )
            global_feature = self.pool(features.transpose(2, 1)).reshape(features.shape[0], -1)
        else:
            raise ValueError(
                "SingleBranchSwin expected 2D, 3D, or 4D features, got {}".format(
                    tuple(features.shape)
                )
            )
        cls, feature = self.classifier(global_feature)
        return [cls, feature]
