# Mask3D ALS 倒树实例分割使用说明

本说明对应仓库内的 `conf/config_fallentree.yaml`。修改保留了 Mask3D 原有模型、loss、Hungarian matcher、inference/post-processing 和 AP evaluator 计算逻辑。新增的 Recall/Precision/F1 仅使用已经完成原有后处理和 inverse mapping、且将送入原 AP evaluator 的同一批最终实例预测。

## 1. 环境

使用已有的 `Mask3D` Conda 环境，不需要重新创建环境：

```bash
conda activate Mask3D
cd /path/to/Mask3D
```

LAS 输出使用 `laspy`；它已写入 `environment.yml`。如当前已有环境尚未安装，只需在该环境中执行：

```bash
pip install "laspy>=2.4,<3"
```

## 2. 原始数据目录和 TXT 格式

从 Mask3D 仓库根目录看，目录应为：

```text
Mask3D/
├── STPLS3D/
│   ├── train/
│   │   ├── Plot_1_Easy.txt
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── conf/
├── datasets/
└── main_instance_segmentation.py
```

每个 TXT 为空白字符分隔的 8 列，不要写表头：

```text
x y z r g b semantic_label instance_id
```

标签约束：

- `semantic_label=0` 表示 `other`，其 `instance_id` 必须为 `-100`；
- `semantic_label=1` 表示 `fallentree`，其 `instance_id` 从 `1` 开始；
- 为了使用 Mask3D 原 AP evaluator 的 `semantic*1000+instance` 编码，单个 plot 的 instance ID 必须小于 `1000`。

预处理脚本会严格检查上述条件，数据不合法时会立即报出文件名和原因。

## 3. 预处理为 `.pth`

在仓库根目录执行：

```bash
python -m datasets.preprocessing.stpls3d_preprocessing preprocess \
  --data_dir="STPLS3D" \
  --save_dir="data/processed/stpls3d_fallentree" \
  --n_jobs=8
```

如果服务器内存较小，可将 `--n_jobs=8` 改为 `--n_jobs=1`。成功后产生：

```text
data/processed/stpls3d_fallentree/
├── train/*.pth
├── validation/*.pth
├── test/*.pth
├── train_database.yaml
├── validation_database.yaml
├── test_database.yaml
├── train_validation_database.yaml
├── label_database.yaml
├── color_mean_std.yaml
└── instance_gt/
    ├── validation/*.txt
    └── test/*.txt
```

`.pth` 内部布局为 Mask3D loader 原有的 12 列：

```text
x' y' z' r g b nx ny nz segment semantic instance
```

`x' y' z'` 仅供稀疏卷积使用，是平移到正坐标范围的坐标。原始 UTM/空间坐标、点数和点顺序始终保留在原 TXT，输出 LAS 时会重新读取原 TXT。

### 标签映射说明

- 外部 semantic 始终为 `0=other, 1=fallentree`；
- 外部背景 instance 始终为 `-100`；
- 只在 Mask3D 内部，预处理将背景 instance 转为原 repo 所用的 no-instance 值 `-1`；
- dataset/collate 使用 `filter_out_classes: [0]` 排除 `other`，因此它不会成为训练 target，也不会进入实例 AP/P/R/F1；
- `label_offset: 1` 将外部类别 `fallentree=1` 映射到模型内部类别 `0`；推理后再由原 loader 的 remap 逻辑映射回 `1`。

## 4. 训练

基本命令：

```bash
python main_instance_segmentation.py \
  --config-name=config_fallentree \
  general.experiment_name=fallentree_run01
```

常用覆盖示例：

```bash
python main_instance_segmentation.py \
  --config-name=config_fallentree \
  general.experiment_name=fallentree_mask3d \
  trainer.max_epochs=1000 \
  data.batch_size=5 \
  data.num_workers=8 \
  data.voxel_size=0.15
```

本配置已设置 `check_val_every_n_epoch: 1`，因此每个 epoch 都运行完整 validation 和原 Mask3D AP evaluator。

### Checkpoint 规则

文件保存在 `saved/<experiment_name>/`：

- 第一个 epoch 同时写入 `last_model.pth` 和 `best_model.pth`；
- 之后每个 epoch 总是覆盖 `last_model.pth`；
- 只在当前原 evaluator `val_mean_ap_50` **严格大于** 历史最佳值时覆盖 `best_model.pth`；
- AP50 相等时不更新 best；
- 两者都是 PyTorch Lightning/Mask3D 完整 checkpoint，包含 `state_dict`、optimizer、scheduler、epoch 和 callback 状态。

以相同 `experiment_name` 重新启动训练时，如果存在 `last_model.pth`，会自动 resume。也可显式指定：

```bash
python main_instance_segmentation.py \
  --config-name=config_fallentree \
  general.experiment_name=fallentree_run01 \
  trainer.resume_from_checkpoint=saved/fallentree_run01/last_model.pth
```

## 5. 训练配置文件与重要参数

主配置是 `conf/config_fallentree.yaml`，它组合了：

- `conf/data/fallentree.yaml`：数据、split、batch size、voxel size；
- `conf/data/datasets/stpls3d_fallentree.yaml`：三个 dataset loader 和标签映射；
- `conf/trainer/trainer_fallentree.yaml`：epoch 与 validation 频率；
- 原 repo 的 `conf/model/mask3d.yaml`、matcher、loss、optimizer 和 scheduler。

与本任务最相关的参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `general.num_targets` | `2` | 1 个有效类 `fallentree` + 1 个 Mask3D no-object 类 |
| `data.num_labels` | `2` | 外部 semantic 类别数：`other` 和 `fallentree` |
| `data.voxel_size` | `0.20` | 稀疏体素尺寸，单位与坐标一致 |
| `data.add_normals` / `data.in_channels` | `true` / `6` | 当前 RGB 全为 0，因此启用预处理中恒为 1 的三个 occupancy/dummy-normal 通道；不改模型结构 |
| `model.num_queries` | `100` | Mask3D object queries；这是原模型参数 |
| `general.topk_per_image` | `100` | 原 inference 保留的 top-k 类别/查询组合 |
| `general.object_score_threshold` | `0.50` | 新增 object P/R/F1 的 prediction confidence 阈值 |
| `general.object_iou_threshold` | `0.50` | 新增 object P/R/F1 的 mask IoU 阈值 |
| `general.save_las` | `true` | test 后是否输出 LAS |
| `general.las_output_dir` | `${general.save_dir}/las_predictions` | LAS 保存目录 |
| `data.train_dataset.filter_out_classes` | `[0]` | 排除 `other` 实例 target |
| `data.train_dataset.label_offset` | `1` | 外部类 `1` 与模型内部类 `0` 之间的映射 |

如果将 processed 数据放在其他位置，需同时修改 `conf/data/datasets/stpls3d_fallentree.yaml` 中 train/validation/test 的 `data_dir`、`label_db_filepath` 和 `color_mean_std`。通常更方便的方式是在命令行同时覆盖三个 dataset 的路径。

## 6. Validation 的 P/R/F1 定义

对每个 plot 独立建立 candidate pairs，要求同类、prediction score 不小于配置阈值且 mask IoU 不小于配置阈值。pairs 按以下顺序 greedy one-to-one matching：

1. mask IoU 从高到低；
2. IoU 相同时 prediction confidence 从高到低；
3. 仍相同时按固定 prediction index、GT index。

各 plot 的 TP/FP/FN 先在整个 validation/test set 累加，再计算 P/R/F1，不会对 plot 指标做算术平均。任一分母为 0 时返回 0。

## 7. Test

用 best checkpoint 运行：

```bash
python main_instance_segmentation.py \
  --config-name=config_fallentree \
  general.train_mode=false \
  general.experiment_name=fallentree_run01 \
  general.checkpoint=saved/fallentree_run01/best_model.pth
```

用其他 checkpoint，只需替换 `general.checkpoint`：

```bash
python main_instance_segmentation.py \
  --config-name=config_fallentree \
  general.train_mode=false \
  general.experiment_name=fallentree_run01 \
  general.checkpoint=/absolute/path/to/another_checkpoint.pth
```

Test 必须保持 `data.test_mode=test`（本配置的默认值）和 `general.export=false`。这里的 `export=false` 只是关闭原 STPLS3D CodaLab TXT export 早退出分支，不会关闭本任务的 `test.log` 或 LAS 输出。

Test 时常用参数：

- `general.checkpoint`：待加载的 `best_model.pth`、`last_model.pth` 或其他兼容 checkpoint；
- `general.object_score_threshold=0.50`：P/R/F1 score 阈值；
- `general.object_iou_threshold=0.50`：P/R/F1 IoU 阈值；
- `general.save_las=true/false`：是否写 LAS；
- `general.las_output_dir=...`：自定义 LAS 目录；
- `general.use_dbscan`、`general.filter_out_instances`、`general.topk_per_image`：原 Mask3D 后处理参数。除非有明确实验设计，建议保持训练/验证与 test 一致。

## 8. 输出位置

以 `general.experiment_name=fallentree_run01` 为例：

```text
saved/fallentree_run01/
├── best_model.pth
├── last_model.pth
├── train.log
├── test.log
├── lightning_logs/
└── las_predictions/
    ├── Plot_9_Easy.las
    └── ...
```

- `train.log`：每个 epoch 追加 validation AP/mAP、AP50、AP25、Recall、Precision 和 F1；
- `test.log`：每个 plot 的 Recall/Precision/F1/TP/FP/FN、TOTAL 累计结果，以及原 evaluator AP/AP50/AP25 与新增指标的组合表；
- `best_model.pth` / `last_model.pth`：完整 Mask3D/Lightning checkpoint；
- `las_predictions/*.las`：每个 test plot 一个。

## 9. LAS 字段和全分辨率映射

每个 LAS 包含原始 TXT 的全部点，点数、顺序和坐标不变。Mask3D 在 voxel 上推理后，使用原 collate 生成的 `inverse_map` 将每个最终 mask 映射回原始点。LAS 的五个 extra dimensions 为：

- `semantic_gt`：原 TXT semantic，`0/1`；
- `semantic_pred`：点所属预测实例的类别，未分配点为 `0`；
- `instance_gt`：原 TXT instance，背景为 `-100`；
- `instance_pred`：从 `1` 开始重新连续编码，未分配点为 `-100`；
- `instance_score`：所属预测实例的 confidence，未分配点为 `0`。

如果原 Mask3D 最终 masks 仍有点级重叠，LAS 唯一归属按 prediction confidence 从高到低处理，同分时按固定 prediction index，因此结果可重复。LAS 使用送入 AP evaluator 的全部最终 predictions；`0.50` score 阈值仅用于按要求计算 P/R/F1，不会改变原 AP 输入。

## 10. 快速检查

不需要 GPU 的 P/R/F1 逻辑测试：

```bash
python -m unittest tests.test_object_metrics -v
```

预处理完成后，建议先用新的 experiment name 运行 1 epoch smoke test：

```bash
python main_instance_segmentation.py \
  --config-name=config_fallentree \
  general.experiment_name=fallentree_smoke \
  trainer.max_epochs=1 \
  data.batch_size=1 \
  data.num_workers=0
```

应同时看到 `last_model.pth`、`best_model.pth` 和包含 validation 指标的 `train.log`。
