# DenseUAV Train Test

这个目录只放分级训练调试入口，不替代根目录的正式训练入口。

约定：

- `Level A`：快速 smoke test
- `Level B`：小数据过拟合
- `Level C`：短程正式训练
- `Level D`：回到根目录执行 `train_test_local.sh`

当前入口：

- `train_progressive_local.sh`

用法：

```bash
cd /home/cjr/GIT_REPO/Compare_Trial/Models/DenseUAV
bash train_test/train_progressive_local.sh
```

你只需要改脚本顶部的 `level` 和常用变量。

设计原则：

- 所有 A/B/C 都调用同一个 `train.py`
- 只通过参数切换训练级别
- 正式训练仍然使用根目录 `train_test_local.sh`

当前补充能力：

- `Level A` 支持 `max_train_batches`
- `Level B` 支持 `max_ids`，可切 `8/16/32`
- 支持 `resume`
- 支持 `latest_checkpoint.pth`
- 支持 `best_checkpoint.pth`
- 支持 `command.txt / runtime_info.txt / pip_freeze.txt / git_commit.txt / git_diff.patch`

推荐流程：

```text
debug_vmamba_forward.py
-> debug_denseuav_with_vmamba.py
-> train_test/train_progressive_local.sh (Level A)
-> train_test/train_progressive_local.sh (Level B)
-> train_test/train_progressive_local.sh (Level C)
-> train_test_local.sh
```
