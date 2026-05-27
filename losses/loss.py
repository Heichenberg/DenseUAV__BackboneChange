import torch
from torch import nn
from .TripletLoss import SameDomainTripletLoss, WeightedSoftTripletLoss, HardMiningTripletLoss, TripletLoss
from .FocalLoss import FocalLoss
import torch.nn.functional as F
from torch.autograd import Variable


def parse_outputs(outputs):
    if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
        cls, feature = outputs
        aux = None
    elif isinstance(outputs, (list, tuple)) and len(outputs) == 3:
        cls, feature, aux = outputs
    else:
        raise ValueError("Unsupported output format for loss: {}".format(type(outputs)))
    return cls, feature, aux


def attention_diversity_loss(attn, eps=1e-6):
    """
    attn: [B, K, N], K is the number of query tokens and N is H*W.
    """
    if attn.ndim != 3:
        raise ValueError("attention_diversity_loss expects [B, K, N], got {}".format(tuple(attn.shape)))
    batch, num_tokens, _ = attn.shape
    attn_norm = F.normalize(attn, p=2, dim=-1, eps=eps)
    sim = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
    mask = ~torch.eye(num_tokens, device=attn.device, dtype=torch.bool)
    off_diag = sim[:, mask]
    return off_diag.mean()


def _new_zero_like(output):
    if isinstance(output, (list, tuple)):
        return output[0].new_tensor(0.0)
    return output.new_tensor(0.0)


class Loss(nn.Module):
    def __init__(self, opt) -> None:
        super(Loss,self).__init__()
        self.opt = opt
        self.diversity_loss_weight = float(getattr(opt, "diversity_loss_weight", 0.0))
        # 分类损失
        if opt.cls_loss == "CELoss":
            self.cls_loss = nn.CrossEntropyLoss()
        elif opt.cls_loss == "FocalLoss":
            self.cls_loss = FocalLoss(alpha=0.25, gamma=2, num_classes = opt.nclasses)
        else:
            self.cls_loss = None

        # 对比损失
        if opt.feature_loss == "TripletLoss":
            self.feature_loss = TripletLoss(margin=0.3, normalize_feature=True)
        elif opt.feature_loss == "HardMiningTripletLoss":
            self.feature_loss = HardMiningTripletLoss(margin=0.3, normalize_feature=True)
        elif opt.feature_loss == "SameDomainTripletLoss":
            self.feature_loss = SameDomainTripletLoss(margin=0.3)
        elif opt.feature_loss == "WeightedSoftTripletLoss":
            self.feature_loss = WeightedSoftTripletLoss()
        elif opt.feature_loss == "ContrastiveLoss":
            from pytorch_metric_learning import losses  # pip install pytorch-metric-learning
            self.feature_loss = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        else:
            self.feature_loss = None

        # KL 损失
        if opt.kl_loss == "KLLoss":
            self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        else:
            self.kl_loss = None
        

    def forward(self, outputs, outputs2, labels, labels2):
        cls1, feature1, aux1 = parse_outputs(outputs)
        cls2, feature2, aux2 = parse_outputs(outputs2)
        loss = 0

        # 分类损失
        res_cls_loss = torch.tensor((0))
        if self.cls_loss is not None:
            res_cls_loss = self.calc_cls_loss(cls1, labels, self.cls_loss) + \
                self.calc_cls_loss(cls2, labels2, self.cls_loss)
            loss += res_cls_loss

        # 特征对比损失
        res_triplet_loss = torch.tensor((0))
        if self.feature_loss is not None:
            split_num = self.opt.batchsize//self.opt.sample_num
            res_triplet_loss = self.calc_triplet_loss(
                feature1, feature2, labels, self.feature_loss, split_num)
            loss += res_triplet_loss

        # 相互学习
        res_kl_loss = torch.tensor((0))
        if self.kl_loss is not None:
            res_kl_loss = self.calc_kl_loss(cls1, cls2, self.kl_loss)
            loss += res_kl_loss

        res_div_loss = _new_zero_like(feature1)
        if self.diversity_loss_weight > 0:
            div_terms = []
            if isinstance(aux1, dict) and "attn" in aux1:
                div_terms.append(attention_diversity_loss(aux1["attn"]))
            if isinstance(aux2, dict) and "attn" in aux2:
                div_terms.append(attention_diversity_loss(aux2["attn"]))
            if div_terms:
                res_div_loss = sum(div_terms) / len(div_terms)
                loss += self.diversity_loss_weight * res_div_loss

        # if self.opt.epoch < self.opt.warm_epoch:
        #     warm_up = 0.1  # We start from the 0.1*lrRate
        #     warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch
        #     warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
        #     loss *= warm_up

        return loss, res_cls_loss, res_triplet_loss, res_kl_loss, res_div_loss
    

    def calc_cls_loss(self, outputs, labels, loss_func):
        loss = 0
        if isinstance(outputs, list):
            for i in outputs:
                loss += loss_func(i, labels)
            loss = loss/len(outputs)
        else:
            loss = loss_func(outputs, labels)
        return loss


    def calc_kl_loss(self, outputs, outputs2, loss_func):
        loss = 0
        if isinstance(outputs, list):
            for i in range(len(outputs)):
                loss += loss_func(F.log_softmax(outputs[i], dim=1),
                                F.softmax(Variable(outputs2[i]), dim=1))
            loss = loss/len(outputs)
        else:
            loss = loss_func(F.log_softmax(outputs, dim=1),
                            F.softmax(Variable(outputs2), dim=1))
        return loss


    def calc_triplet_loss(self, outputs, outputs2, labels, loss_func, split_num=8):
        if isinstance(outputs, list):
            loss = 0
            for i in range(len(outputs)):
                out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
                labels_concat = torch.cat((labels, labels), dim=0)
                loss += loss_func(out_concat, labels_concat)
            loss = loss/len(outputs)
        else:
            out_concat = torch.cat((outputs, outputs2), dim=0)
            labels_concat = torch.cat((labels, labels), dim=0)
            loss = loss_func(out_concat, labels_concat)
        return loss
