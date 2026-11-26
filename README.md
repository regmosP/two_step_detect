# 饼干缺陷检测

面向物理测试台图像的双阶段 YOLO11 缺陷检测项目，包含预处理、全局候选筛查、难例专模复核和测试时增强（TTA）框融合。

## 项目结构

```text
.
├── cookie_defect/
│   ├── __init__.py
│   ├── pipeline.py       # 双阶段检测与 TTA
│   └── preprocessing.py  # 光照/噪声预处理
├── infer.py              # 图片/目录推理入口
├── train.py              # YOLO11 训练入口
├── prepare_specialist.py # 从标注生成难例专模裁剪集
├── preprocess_dataset.py # 为预处理消融生成同分布训练数据
├── calibrate_thresholds.py # 用留出集扫描各类别置信度阈值
└── requirements.txt
```

## 训练

先按 Ultralytics YOLO 格式准备标注数据集（`images/train`、`images/val`、`labels/train`、`labels/val` 和数据 YAML）。全局模型使用完整缺陷类别数据训练：

```bash
python train.py --data data/global.yaml --weights yolo11m.pt --name cookie-global --imgsz 960 --epochs 200 --batch 16
```

从同一份 YOLO 标注集提取“缺角”“破碎”目标及周边上下文，建立专模裁剪数据集。类别名称按原始 YAML 的 class id 顺序传入；脚本按源图像划分验证集，避免同一图像的裁剪同时进入训练和验证：

```bash
python prepare_specialist.py --images data/images --labels data/labels \
  --class-names "完整,缺角,破碎,污点" --target-classes "缺角,破碎" \
  --output data/specialist
python train.py --data data/specialist/data.yaml --weights yolo11m.pt \
  --name cookie-hard-defects --imgsz 960 --epochs 200 --batch 16
```

训练默认采用预训练 YOLO11m、较高输入分辨率、AdamW、余弦学习率、早停、后期关闭 Mosaic，以及轻量颜色/几何增强。显存不足时降低 `--batch` 或 `--imgsz`；目标很小且显存允许时可升至 1280。默认推理使用原始图，和训练输入保持一致；选择非 raw 预处理时，必须先对训练、验证图像应用相同处理再训练，不能只在推理时更换图像风格。应以按采集批次/日期留出的验证集检查 mAP50-95、每类召回率、混淆矩阵和误报/漏报案例，避免同一饼干的近似连拍跨越训练集与验证集。

## 阈值校准与预处理消融

先用低输出阈值保存尽可能完整的预测，再用独立留出集扫描每类置信度阈值。校准集不要同时作为最终报告性能的测试集：

```bash
python infer.py --image data/images/val --global-weights runs/train/cookie-global/weights/best.pt \
  --specialist-weights runs/train/cookie-hard-defects/weights/best.pt --output runs/val-low \
  --conf 0.05 --global-candidate-conf 0.05 --specialist-conf 0.05 --preprocess raw
python calibrate_thresholds.py --predictions runs/val-low --images data/images/val \
  --labels data/labels/val --class-names "完整,缺角,破碎,污点" --output runs/thresholds.csv
```

脚本按 IoU 匹配计算每类 precision、recall 和 F1，并输出扫描表。可将挑选后的值传给 `infer.py`，例如 `--class-conf "缺角:0.32,破碎:0.27"`。阈值目标应服从产线代价：漏检代价高时优先选高召回点，误报复检代价高时优先选高精度点。

可在同一验证集上用 `--preprocess raw|clahe|binary|enhanced` 比较输入处理。不同模式各自产生的预测需要分别校准阈值；如非 raw 模式胜出，应将同样的预处理应用于训练和验证图片后重新训练，再对最终模型校准。当前流水线使用 TTA 视图的置信度加权平均，避免视图数量增加时置信度机械升高。

为非 raw 模式生成与推理一致的数据集（源标签目录需与图片目录的 train/val 子目录对应）：

```bash
python preprocess_dataset.py --images data/images --labels data/labels \
  --output data/clahe --mode clahe --class-names "完整,缺角,破碎,污点"
python train.py --data data/clahe/data.yaml --name cookie-global-clahe
```

现场采集建议固定相机距离、曝光、白平衡和饼干摆放；尽量减少反光与变化阴影。训练样本要覆盖生产环境的光照、碎屑和背景变化，并把模型误报的正常边缘、纹理和阴影整理成专模负样本。提高输入分辨率或重叠切片只对小缺陷/高分辨率图有效，须同时比较推理耗时和缺陷召回率。

## 安装与使用

```bash
python -m pip install -r requirements.txt
python infer.py --image path/to/image.jpg \
  --global-weights weights/global.pt \
  --specialist-weights weights/difficult.pt \
  --output runs/predictions
```

难例类别默认按名称匹配 `缺角`、`破碎`。如果模型使用英文或其他类别名，请传入 `--specialist-classes`，以逗号分隔。专模类别通过 `--specialist-class-map` 映射至全局模型类别，格式为 `专模类别名:全局类别名`，例如 `缺角:缺角,破碎:破碎`。专模仅复核第一阶段对应候选区域；其余类别沿用全局模型结果。

输入图像默认保持原样（与训练输入一致）；`--preprocess` 可切换 `raw`、`clahe`、`binary`、`enhanced` 做对照。TTA 默认使用原尺度、水平翻转和 0.85/1.15 尺度，预测框映射回原图后按类别做 IoU 聚类和坐标/置信度加权融合。`--global-candidate-conf`、`--specialist-conf` 控制前两阶段的候选门槛，`--conf` 或 `--class-conf` 控制最终输出门槛。

## 权重和数据

权重文件与饼干数据不包含在代码仓库中。请将训练好的 Ultralytics YOLO11 `.pt` 权重放在 `weights/`，或指定其他路径。建议分别训练全局模型和以缺角、破碎难例为重点的专精模型；模型类别名称应与命令行映射一致。

程序输出带检测框的图片及同名 JSON。JSON 坐标为原始输入图上的像素坐标，格式为 `xyxy`。
