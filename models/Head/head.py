import torch.nn as nn
from .SingleBranch import SingleBranch, SingleBranchCNN, SingleBranchSwin
from .FSRA import FSRA, FSRA_CNN
from .LPN import LPN, LPN_CNN
from .GeM import GeM
from .NetVLAD import NetVLAD
from .MixerHead import MixerHeadV1
from .GeoTokenHead import GeoTokenHeadV1
from .GeoTokenV2Head import GeoTokenV2Head
from .GeoTokenV2CAGHead import GeoTokenV2CAGHead
from .GeoTokenV2CAGMixerHead import (
    GeoTokenV2CAGMixer2Head,
    GeoTokenV2CAGMixer4Head,
    GeoTokenV2CAGMixer8Head,
)
from .GeoTokenV2MixerHead import GeoTokenV2Mixer1Head, GeoTokenV2Mixer2Head
from .MSGE_GeoTokenV2 import MSGE_GeoTokenV2
from .MSGE_GeoTokenV2CAGHead import MSGE_GeoTokenV2CAGHead
from .MSGE_GeoTokenV2CAGDivHead import MSGE_GeoTokenV2CAGDivHead
from .MSGE_GeoTokenV2CAGMixerHead import (
    MSGE_GeoTokenV2CAGMixer1Head,
    MSGE_GeoTokenV2CAGMixer2Head,
    MSGE_GeoTokenV2CAGMixer4Head,
)

def make_head(opt):
    return Head(opt)


class Head(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.head = self.init_head(opt)
        self.opt = opt

    def init_head(self, opt):
        head = opt.head
        if head == "SingleBranch":
            head_model = SingleBranch(opt)
        elif head == "SingleBranchCNN":
            head_model = SingleBranchCNN(opt)
        elif head == "SingleBranchSwin":
            head_model = SingleBranchSwin(opt)
        elif head == "NetVLAD":
            head_model = NetVLAD(opt)
        elif head == "FSRA":
            head_model = FSRA(opt)
        elif head == "FSRA_CNN":
            head_model = FSRA_CNN(opt)
        elif head == "LPN":
            head_model = LPN(opt)
        elif head == "LPN_CNN":
            head_model = LPN_CNN(opt)
        elif head == "GeM":
            head_model = GeM(opt)
        elif head == "MixerHeadV1":
            head_model = MixerHeadV1(opt)
        elif head == "GeoTokenHeadV1":
            head_model = GeoTokenHeadV1(opt)
        elif head == "GeoTokenV2Head":
            head_model = GeoTokenV2Head(opt)
        elif head == "GeoTokenV2CAGHead":
            head_model = GeoTokenV2CAGHead(opt)
        elif head == "GeoTokenV2CAGMixer2Head":
            head_model = GeoTokenV2CAGMixer2Head(opt)
        elif head == "GeoTokenV2CAGMixer4Head":
            head_model = GeoTokenV2CAGMixer4Head(opt)
        elif head == "GeoTokenV2CAGMixer8Head":
            head_model = GeoTokenV2CAGMixer8Head(opt)
        elif head == "GeoTokenV2Mixer1Head":
            head_model = GeoTokenV2Mixer1Head(opt)
        elif head == "GeoTokenV2Mixer2Head":
            head_model = GeoTokenV2Mixer2Head(opt)
        elif head == "MSGE_GeoTokenV2":
            head_model = MSGE_GeoTokenV2(opt)
        elif head == "MSGE_GeoTokenV2CAGHead":
            head_model = MSGE_GeoTokenV2CAGHead(opt)
        elif head == "MSGE_GeoTokenV2CAGDivHead":
            head_model = MSGE_GeoTokenV2CAGDivHead(opt)
        elif head == "MSGE_GeoTokenV2CAGMixer1Head":
            head_model = MSGE_GeoTokenV2CAGMixer1Head(opt)
        elif head == "MSGE_GeoTokenV2CAGMixer2Head":
            head_model = MSGE_GeoTokenV2CAGMixer2Head(opt)
        elif head == "MSGE_GeoTokenV2CAGMixer4Head":
            head_model = MSGE_GeoTokenV2CAGMixer4Head(opt)
        else:
            raise NameError("{} not in the head list!!!".format(head))
        return head_model

    def forward(self, features):
        features = self.head(features)
        return features
