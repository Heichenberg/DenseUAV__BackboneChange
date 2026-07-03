# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import time
from optimizers.make_optimizer import make_optimizer
# from models.model import make_model
from models.taskflow import make_model
from datasets.make_dataloader import make_dataset
from datasets.dss_fss import build_fss_neighbors, should_update_fss
from datasets.dss_stage import DSSStageController
from tool.utils import save_network, copyfiles2checkpoints, get_preds, get_logger, calc_flops_params, set_seed, save_training_checkpoint, load_training_checkpoint, save_run_artifacts, move_optimizer_state
import warnings
from losses.loss import Loss


warnings.filterwarnings("ignore")


def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default='test',
                        type=str, help='the experiment name that will be saved in checkpoints dir in the root')
    parser.add_argument('--data_dir', default='/home/dmmm/Dataset/DenseUAV/data_2022/train',
                        type=str, help='training dir path')
    parser.add_argument('--num_worker', default=0, type=int, help='')
    parser.add_argument('--batchsize', default=2, type=int, help='batchsize')
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--h', default=224, type=int, help='height')
    parser.add_argument('--w', default=224, type=int, help='width')
    parser.add_argument('--rr', default="", type=str, help='random rotate')
    parser.add_argument('--ra', default="", type=str, help='random affine')
    parser.add_argument('--re', default="", type=str, help='random erasing')
    parser.add_argument('--cj', default="", type=str, help='color jitter')
    parser.add_argument('--disable_hflip', action='store_true',
                        help='disable default random horizontal flip in training transforms')
    parser.add_argument('--erasing_p', default=0.3, type=float,
                        help='random erasing probability, in [0,1]')
    parser.add_argument('--warm_epoch', default=0, type=int,
                        help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--backbone_lr', default=0.0, type=float,
                        help='learning rate for backbone param group; <=0 means use --lr')
    parser.add_argument('--head_lr', default=0.0, type=float,
                        help='learning rate for head / non-backbone param group; <=0 means use --lr')
    parser.add_argument('--DA', action='store_true',
                        help='use Color Data Augmentation')
    parser.add_argument('--droprate', default=0.5,
                        type=float, help='drop rate')
    parser.add_argument('--autocast', action='store_true',
                        default=True, help='use mix precision')
    parser.add_argument('--disable_autocast', action='store_true',
                        help='disable mixed precision for debug runs')
    parser.add_argument('--block', default=2, type=int, help='')
    parser.add_argument('--cls_loss', default="CELoss", type=str, help='loss type of representation learning')
    parser.add_argument('--feature_loss', default="no", type=str, help='loss type of metric learning')
    parser.add_argument('--kl_loss', default="no", type=str, help='loss type of mutual learning')
    parser.add_argument('--sample_num', default=1, type=int,
                        help='num of repeat sampling')
    parser.add_argument('--train_strategy', default='origin', type=str, choices=['origin', 'dss'],
                        help='training sampler strategy')
    parser.add_argument('--dss_gps_file', default="", type=str,
                        help='GPS file used by DSS GDS sampling; defaults to <data_root>/Dense_GPS_train.txt')
    parser.add_argument('--dss_start_epoch', default=0, type=int,
                        help='epoch to start DSS sampling')
    parser.add_argument('--dss_gds_topk', default=64, type=int,
                        help='number of nearest geographic negative IDs stored per anchor')
    parser.add_argument('--dss_gds_ratio', default=0.5, type=float,
                        help='DSS v1 ratio for GDS negatives inside the non-anchor part of a batch')
    parser.add_argument('--dss_fss_ratio', default=0.0, type=float,
                        help='DSS ratio for FSS negatives; ignored until fss_neighbors is populated')
    parser.add_argument('--dss_fss_topk', default=64, type=int,
                        help='number of nearest feature negative IDs stored per anchor')
    parser.add_argument('--dss_fss_start_epoch', default=10, type=int,
                        help='0-based epoch to start refreshing DSS FSS neighbors')
    parser.add_argument('--dss_fss_samples_per_id', default=1, type=int,
                        help='number of random satellite-drone pairs sampled per ID when building FSS')
    parser.add_argument('--dss_rs_ratio', default=0.5, type=float,
                        help='DSS v1 ratio for random IDs inside the non-anchor part of a batch')
    parser.add_argument('--dss_fss_update_interval', default=10, type=int,
                        help='planned interval for refreshing FSS neighbors')
    parser.add_argument('--dss_cache_dir', default="", type=str,
                        help='cache dir for dataset-level DSS files; defaults to <DenseUAV>/dss_cache')
    parser.add_argument('--dss_stage_mode', default='fixed', type=str, choices=['fixed', 'loss_adaptive'],
                        help='DSS stage schedule mode')
    parser.add_argument('--dss_ce_threshold', default=2.0, type=float,
                        help='CE EMA threshold for switching DSS early -> middle')
    parser.add_argument('--dss_plateau_delta', default=0.05, type=float,
                        help='relative total-loss EMA drop below this value counts as plateau')
    parser.add_argument('--dss_plateau_patience', default=3, type=int,
                        help='number of plateau epochs for switching DSS middle -> late')
    parser.add_argument('--dss_ema_momentum', default=0.9, type=float,
                        help='EMA momentum for DSS adaptive stage losses')
    parser.add_argument('--num_epochs', default=120, type=int, help='total epoches for training')
    parser.add_argument('--num_bottleneck', default=512, type=int, help='the dimensions for embedding the feature')
    parser.add_argument('--load_from', default="", type=str, help='checkpoints path for pre-loading')
    parser.add_argument('--backbone', default="cvt13", type=str, help='backbone network for applying')
    parser.add_argument('--backbone_weight', default="", type=str, help='pretrained backbone checkpoint path')
    parser.add_argument('--head', default="FSRA_CNN", type=str, help='head type for applying')
    parser.add_argument('--head_pool', default="max", type=str, help='head pooling type for applying')
    parser.add_argument('--max_train_batches', default=0, type=int,
                        help='max batches per epoch, 0 means no limit')
    parser.add_argument('--max_total_batches', default=0, type=int,
                        help='max batches for the whole run, 0 means no limit')
    parser.add_argument('--max_ids', default=0, type=int,
                        help='use only the first N sorted train IDs, 0 means all')
    parser.add_argument('--id_subset_file', default="", type=str,
                        help='text file containing one train ID per line')
    parser.add_argument('--seed', default=666, type=int, help='random seed')
    parser.add_argument('--resume', default="", type=str, help='resume training checkpoint path')
    parser.add_argument('--eval_interval', default=1, type=int, help='epoch interval for latest/best checkpoint update')
    parser.add_argument('--save_latest', action='store_true', default=True, help='save latest checkpoint during training')
    parser.add_argument('--save_best', action='store_true', default=True, help='save best checkpoint during training')
    parser.add_argument('--best_metric', default='satellite_acc', type=str, choices=['satellite_acc', 'drone_acc', 'loss'], help='metric used for best checkpoint')
    

    opt = parser.parse_args()
    if opt.disable_autocast:
        opt.autocast = False
    if opt.backbone_lr <= 0:
        opt.backbone_lr = opt.lr
    if opt.head_lr <= 0:
        opt.head_lr = opt.lr
    print(opt)
    return opt

def metric_value(summary, metric_name):
    if metric_name == "loss":
        return -summary["epoch_loss"]
    if metric_name == "drone_acc":
        return summary["epoch_acc2"]
    return summary["epoch_acc"]


def get_context_gate_value(model):
    model_ref = model.module if hasattr(model, "module") else model
    for module in model_ref.modules():
        if module is model_ref:
            continue
        if hasattr(module, "get_context_gate_value"):
            return module.get_context_gate_value()
    return None


def write_context_gate_file(opt, model, best_metric=None):
    gate_value = get_context_gate_value(model)
    gate_path = os.path.join("checkpoints", opt.name, "context_gate.txt")
    with open(gate_path, "w") as f:
        f.write("context_gate={}\n".format("None" if gate_value is None else "{:.8f}".format(gate_value)))
        f.write("best_metric={}\n".format(best_metric))


def apply_dss_ratios(opt, sampler, ratios):
    if not hasattr(sampler, "set_dss_ratios"):
        return
    sampler.set_dss_ratios(
        gds_ratio=ratios["gds_ratio"],
        fss_ratio=ratios["fss_ratio"],
        rs_ratio=ratios["rs_ratio"],
    )
    opt.dss_gds_ratio = ratios["gds_ratio"]
    opt.dss_fss_ratio = ratios["fss_ratio"]
    opt.dss_rs_ratio = ratios["rs_ratio"]


def train_model(model, opt, optimizer, scheduler, dataloaders, dataset_sizes, start_epoch=0, best_metric=None):
    logger = get_logger(
        "checkpoints/{}/train.log".format(opt.name))

    # thop计算MACs
    # macs, params = calc_flops_params(
    #     model, (1, 3, opt.h, opt.w), (1, 3, opt.h, opt.w))
    # logger.info("model MACs={}, Params={}".format(macs, params))

    use_gpu = opt.use_gpu
    num_epochs = opt.num_epochs
    since = time.time()
    scaler = GradScaler(enabled=opt.autocast and use_gpu)
    nnloss = Loss(opt)
    if use_gpu:
        current_device = torch.cuda.current_device()
        logger.info(
            "CUDA enabled: device={} name={} autocast={}".format(
                current_device, torch.cuda.get_device_name(current_device), opt.autocast
            )
        )
    else:
        logger.info("CUDA disabled: training on CPU")
    total_batches_seen = 0
    dss_stage_controller = None
    if getattr(opt, "train_strategy", "origin") == "dss" and getattr(opt, "dss_stage_mode", "fixed") == "loss_adaptive":
        dss_stage_controller = DSSStageController(opt)
        apply_dss_ratios(opt, dataloaders.sampler, dss_stage_controller.current_ratios())
        logger.info("DSS adaptive stage initialized: stage=early ratios={}".format(dss_stage_controller.current_ratios()))
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 50)

        running_cls_loss = 0.0
        running_triplet = 0.0
        running_kl_loss = 0.0
        running_loss = 0.0
        running_corrects = 0.0
        running_corrects2 = 0.0
        epoch_batches = 0
        epoch_samples = 0
        stop_training = False
        if hasattr(dataloaders.sampler, "set_epoch"):
            dataloaders.sampler.set_epoch(epoch)
        sampler_fss_ratio = getattr(dataloaders.sampler, "fss_ratio", getattr(opt, "dss_fss_ratio", 0.0))
        has_fss_neighbors = getattr(dataloaders.sampler, "fss_neighbors", None) is not None
        if hasattr(dataloaders.sampler, "update_fss_neighbors") and should_update_fss(
            epoch, opt, fss_ratio=sampler_fss_ratio, has_neighbors=has_fss_neighbors
        ):
            fss_neighbors = build_fss_neighbors(model, dataloaders.dataset, opt, logger=logger, epoch=epoch)
            dataloaders.sampler.update_fss_neighbors(fss_neighbors)
            opt._last_fss_update_epoch = epoch

        model.train(True)  # Set model to training mode
        for data, data3 in dataloaders:
            if opt.max_train_batches > 0 and epoch_batches >= opt.max_train_batches:
                logger.info('Reached max_train_batches=%d for epoch %d', opt.max_train_batches, epoch)
                break
            if opt.max_total_batches > 0 and total_batches_seen >= opt.max_total_batches:
                logger.info('Reached max_total_batches=%d, stopping training early', opt.max_total_batches)
                stop_training = True
                break
            # 获取输入无人机和卫星数据
            inputs, labels = data
            inputs3, labels3 = data3
            now_batch_size = inputs.shape[0]
            if now_batch_size < opt.batchsize:  # skip the last batch
                continue
            if use_gpu:
                inputs = Variable(inputs.cuda().detach())
                inputs3 = Variable(inputs3.cuda().detach())
                labels = Variable(labels.cuda().detach())
                labels3 = Variable(labels3.cuda().detach())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # 梯度清零
            optimizer.zero_grad()

            # start_time = time.time()
            # 模型前向传播
            with autocast(enabled=opt.autocast and use_gpu):
                outputs, outputs2 = model(inputs, inputs3)
            # print("model_time:{}".format(time.time()-start_time))
            # 计算损失
            loss, cls_loss, f_triplet_loss, kl_loss = nnloss(
                outputs, outputs2, labels, labels3)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss detected: {}".format(loss.item()))
            # start_time = time.time()
            # 反向传播
            if opt.autocast:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            # print("backward_time:{}".format(time.time()-start_time))

            # 统计损失
            running_loss += loss.item() * now_batch_size
            running_cls_loss += cls_loss.item()*now_batch_size
            running_triplet += f_triplet_loss.item() * now_batch_size
            running_kl_loss += kl_loss.item() * now_batch_size
            epoch_batches += 1
            epoch_samples += now_batch_size
            total_batches_seen += 1

            # 统计精度
            preds, preds2 = get_preds(outputs[0], outputs2[0])
            if isinstance(preds, list) and isinstance(preds2, list):
                running_corrects += sum([float(torch.sum(pred == labels.data))
                                        for pred in preds])/len(preds)
                running_corrects2 += sum([float(torch.sum(pred == labels3.data))
                                         for pred in preds2]) / len(preds2)
            else:
                running_corrects += float(torch.sum(preds == labels.data))
                running_corrects2 += float(torch.sum(preds2 == labels3.data))

        # 统计损失和精度
        denom = max(epoch_samples, 1)
        epoch_cls_loss = running_cls_loss / denom
        epoch_kl_loss = running_kl_loss / denom
        epoch_triplet_loss = running_triplet / denom
        epoch_loss = running_loss / denom
        epoch_acc = running_corrects / denom
        epoch_acc2 = running_corrects2 / denom

        lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']
        lr_other = optimizer.state_dict()['param_groups'][1]['lr']
        logger.info('Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f} actual_num_batches:{} actual_num_samples:{}'
                    .format(epoch_loss, epoch_cls_loss, epoch_kl_loss,
                            epoch_triplet_loss, epoch_acc,
                            epoch_acc2, lr_backbone, lr_other, epoch_batches, epoch_samples))
        if hasattr(dataloaders.sampler, "last_stats") and dataloaders.sampler.last_stats:
            logger.info("DSS sampler stats: {}".format(dataloaders.sampler.last_stats))
        if dss_stage_controller is not None:
            stage_info = dss_stage_controller.update(epoch, epoch_cls_loss, epoch_loss)
            apply_dss_ratios(opt, dataloaders.sampler, stage_info)
            if stage_info["changed"]:
                logger.info(
                    "DSS stage changed at epoch %d: %s -> %s ce_ema=%.4f total_loss_ema=%.4f loss_drop=%s ratios={'gds': %.3f, 'fss': %.3f, 'rs': %.3f}",
                    epoch,
                    stage_info["old_stage"],
                    stage_info["stage"],
                    stage_info["ce_ema"],
                    stage_info["total_loss_ema"],
                    "None" if stage_info["loss_drop"] is None else "{:.6f}".format(stage_info["loss_drop"]),
                    stage_info["gds_ratio"],
                    stage_info["fss_ratio"],
                    stage_info["rs_ratio"],
                )
            else:
                logger.info(
                    "DSS stage state: epoch=%d stage=%s ce_ema=%.4f total_loss_ema=%.4f loss_drop=%s plateau_count=%d ratios={'gds': %.3f, 'fss': %.3f, 'rs': %.3f}",
                    epoch,
                    stage_info["stage"],
                    stage_info["ce_ema"],
                    stage_info["total_loss_ema"],
                    "None" if stage_info["loss_drop"] is None else "{:.6f}".format(stage_info["loss_drop"]),
                    stage_info["plateau_count"],
                    stage_info["gds_ratio"],
                    stage_info["fss_ratio"],
                    stage_info["rs_ratio"],
                )
        context_gate = get_context_gate_value(model)
        if context_gate is not None:
            logger.info("GeoToken context_gate: {:.4f}".format(context_gate))

        scheduler.step()
        summary = {
            "epoch_loss": epoch_loss,
            "epoch_cls_loss": epoch_cls_loss,
            "epoch_kl_loss": epoch_kl_loss,
            "epoch_triplet_loss": epoch_triplet_loss,
            "epoch_acc": epoch_acc,
            "epoch_acc2": epoch_acc2,
            "lr_backbone": lr_backbone,
            "lr_other": lr_other,
            "epoch_batches": epoch_batches,
            "epoch_samples": epoch_samples,
        }
        if opt.eval_interval > 0 and ((epoch + 1) % opt.eval_interval == 0 or epoch == num_epochs - 1 or stop_training):
            current_metric = metric_value(summary, opt.best_metric)
            if opt.save_best and (best_metric is None or current_metric > best_metric):
                best_metric = current_metric
                save_training_checkpoint(model, optimizer, scheduler, opt.name, "best_checkpoint.pth", epoch, best_metric)
            if opt.save_latest:
                save_training_checkpoint(model, optimizer, scheduler, opt.name, "latest_checkpoint.pth", epoch, best_metric)
        if epoch % 10 == 9 and epoch >= 110:
            save_network(model, opt.name, epoch)

        time_elapsed = time.time() - since
        since = time.time()
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        if stop_training:
            break
    write_context_gate_file(opt, model, best_metric)
    return best_metric


if __name__ == '__main__':
    opt = get_parse()
    set_seed(opt.seed)
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    use_gpu = torch.cuda.is_available()
    opt.use_gpu = use_gpu
    # set gpu ids
    if use_gpu and len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    dataloaders, class_names, dataset_sizes = make_dataset(opt)
    opt.nclasses = len(class_names)

    checkpoint_dir = copyfiles2checkpoints(opt)
    save_run_artifacts(opt, checkpoint_dir)

    model = make_model(opt)

    optimizer_ft, exp_lr_scheduler = make_optimizer(model, opt)
    start_epoch = 0
    best_metric = None
    if opt.resume and os.path.exists(opt.resume):
        checkpoint = load_training_checkpoint(opt.resume, model, optimizer_ft, exp_lr_scheduler, map_location='cpu')
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_metric = checkpoint.get("best_metric")
        print("Resumed training from:", opt.resume)
        print("start_epoch:", start_epoch)
        print("best_metric:", best_metric)

    if use_gpu:
        model = model.cuda()
        move_optimizer_state(optimizer_ft, torch.device("cuda", gpu_ids[0] if len(gpu_ids) > 0 else 0))

    train_model(model, opt, optimizer_ft, exp_lr_scheduler,
                dataloaders, dataset_sizes, start_epoch=start_epoch, best_metric=best_metric)
