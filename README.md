# CHOIR

CHOIR 代码按处理阶段组织：

```text
CHOIR/
├── data/                       # 本地输入，不提交到 Git
│   └── <VIDEO_ID>.mp4
├── output/                     # 标注、可视化及处理结果，不提交到 Git
│   └── <VIDEO_ID>/
├── stage1/
│   ├── Yolov8/
│   ├── sam-3d-objects/
│   └── Dyn_HaMR_new/
├── stage2/
│   └── DexGraspNet_table/
├── diffusion-vas/
└── GraspFlowMatching/
```

除 README 明确列出的两个 Yolov8 detector checkpoint 外，其他模型权重、
输入数据、运行缓存和输出结果不提交到 Git，请按照各模块说明单独下载。

## Python 环境总览

除 `stage1/Dyn_HaMR_new` 外，仓库中的其他代码统一使用 `hd` 环境
（含 `sam-3d-objects`；推理请在有 GPU 的 worker 上跑）：

```text
/vepfs_default/chanxueyan/lhp/xh/env/hd
```

覆盖的模块包括：

- `stage1/Yolov8`
- `stage1/sam-3d-objects`
- `stage2/DexGraspNet_table`
- `diffusion-vas`
- `GraspFlowMatching`

激活方式：

```bash
conda activate /vepfs_default/chanxueyan/lhp/xh/env/hd
```

当前机器上验证过的主要版本：

| Python | PyTorch | TorchVision | CUDA runtime |
|---:|---:|---:|---:|
| 3.10.18 | 2.6.0+cu126 | 0.21.0+cu126 | 12.6 |

部分关键包版本：

```text
ultralytics==8.3.220
opencv-python==4.12.0.88
hydra-core==1.3.2
diffusers==0.35.1
transformers==4.56.2
accelerate==1.11.0
```

验证环境：

```bash
/vepfs_default/chanxueyan/lhp/xh/env/hd/bin/python -c \
  "import torch; print(torch.__version__, torch.version.cuda)"
```

## Stage 1：Yolov8 手和物体预处理

### 功能

从 RGB 视频中检测并跟踪左右手，再用 SAM2 生成并传播手部和交互物体的
分割掩码。输出供 `stage1/sam-3d-objects` 及后续手物交互重建使用。

两条推理入口的核心区别是物体初始提示的来源：

| 入口 | 检测模型 | 物体初始提示 | 适用场景 |
|---|---|---|---|
| `run_with_manual_object_mask.py` | `models/wilor_hand_detector.pt` | `output/<VIDEO_ID>/annotations/first_mask.png` | 人工初始物体 mask |
| `run_with_hoi_detector.py` | `models/tasterob_hoi_detector.pt` | 模型检测到的 object bbox | 自动检测手和交互物体 |

两条入口都会：检测左右手 bbox 与 21 个关键点、用 ByteTrack 维持轨迹、用
SAM2 传播手/物体 mask，并导出 RGB、mask、bbox 与关键点。

### 目录结构

```text
stage1/Yolov8/
├── run_with_manual_object_mask.py   # 流水线 A：外部 first_mask
├── run_with_hoi_detector.py         # 流水线 B：HOI detector
├── labeling.py                      # 流水线 A 的交互式标注
├── data_layout.py                   # 共享 data/output 路径约定
├── models/
│   ├── wilor_hand_detector.pt       # hand-only detector
│   └── tasterob_hoi_detector.pt     # left / right / object 三类 detector
├── configs/
│   └── bytetrack.yaml
├── sam2/                            # 本地 SAM2 包（不含权重）
├── training/                        # 伪标签与微调脚本
└── tests/
```

仓库根目录的输入输出约定：

```text
data/<VIDEO_ID>.mp4                  # 仅原始输入视频（平铺）
output/<VIDEO_ID>/                   # 全部派生结果
├── <VIDEO_ID>.mp4                   # 处理后视频（A 为裁剪后，B 为复制）
├── annotations/                     # labeling.py 生成的人工标注
├── visualizations/                  # 推理调试可视化
├── rgbs/
├── obj_masks/
├── rh_masks/
├── lh_masks/
└── *.json
```

`data/` 与 `output/` 已由根目录 `.gitignore` 排除。训练伪标签默认写到
`stage1/Yolov8/data/` 与 `stage1/Yolov8/data_hoi/`，与根目录输入分离。

### 环境与依赖

使用 `hd` 环境，并在 `stage1/Yolov8` 目录下运行（相对路径依赖本地
`sam2/` 与 `configs/bytetrack.yaml`）：

```bash
conda activate /vepfs_default/chanxueyan/lhp/xh/env/hd
cd /vepfs_default/chanxueyan/lhp/xh/code/CHOIR-upload/stage1/Yolov8
```

主要依赖：`torch`、`torchvision`、`ultralytics`、`opencv-python`、
`imageio`、`imageio-ffmpeg`、`numpy`、`tqdm`、`hydra-core`、
`omegaconf`、`iopath`、`Pillow`。系统还需 `ffmpeg`。

两个 detector checkpoint 已包含在仓库中。SAM2.1 Large 权重需单独准备：

```text
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

当前两个推理入口暂时从本机绝对路径加载该权重：

```text
/vepfs_default/chanxueyan/lhp/xh/code/sam2-main/checkpoints/ckpts/sam2.1_hiera_large.pt
```

`labeling.py` 默认查找 `stage1/Yolov8/sam2/checkpoints/sam2.1_hiera_large.pt`，
也可通过 `--ckpt` 显式指定。

### 流水线 A：外部物体 mask

#### 1. 用 labeling.py 标注

在原始视频第 0 帧交互式标注物体，将 bbox / 正负点转为
`first_mask.png`。默认不做全视频 tracking。

```bash
python labeling.py \
  --data /vepfs_default/chanxueyan/lhp/xh/code/CHOIR-upload/data \
  --output /vepfs_default/chanxueyan/lhp/xh/code/CHOIR-upload/output \
  --video-id demo \
  --ckpt /vepfs_default/chanxueyan/lhp/xh/code/sam2-main/checkpoints/ckpts/sam2.1_hiera_large.pt
```

需要图形界面或 X11 转发。交互：`b`/`p` 切换 BBox/Point；`Enter`/`Space`
预览；`s` 保存；`r` 重置；`n` 跳过；`q` 退出。

保存结果：

```text
output/<VIDEO_ID>/annotations/
├── frame_000000.jpg
├── first_mask.png
├── first_mask_overlay.jpg
└── prompt.json
```

#### 2. 运行检测与 tracking

```bash
python run_with_manual_object_mask.py \
  --video_id <VIDEO_ID> \
  --data_dir /vepfs_default/chanxueyan/lhp/xh/code/CHOIR-upload/data \
  --output_dir /vepfs_default/chanxueyan/lhp/xh/code/CHOIR-upload/output \
  --model models/wilor_hand_detector.pt \
  --num_workers 1 \
  --conf 0.5
```

说明：

- 省略 `--video_id` 时处理 `--data_dir` 下全部 MP4。
- `--data_dir` / `--output_dir` 默认已指向仓库根目录 `data/`、`output/`。
- 建议先用 `--num_workers 1`；每 worker 都会加载 YOLO 与 SAM2 Large。
- `--no_visualize` 可跳过 `output/<VIDEO_ID>/visualizations/`。
- 流水线 A 会裁剪手部完整入镜时段，并将帧统一为 1920×1080。
- `first_mask.png` 必须对应裁剪前原始视频的第 0 帧。

### 流水线 B：自动检测交互物体

`models/tasterob_hoi_detector.pt` 在 WiLoR hand detector 基础上，于
Taste-Rob 数据上微调得到三类（left / right / object）YOLO Pose 模型。
脚本收集 object bbox 候选，按置信度用 SAM2 生成初始物体 mask，再向整段
视频传播。

```bash
python run_with_hoi_detector.py \
  --data_source local \
  --video_id <VIDEO_ID> \
  --data_dir /vepfs_default/chanxueyan/lhp/xh/code/CHOIR-upload/data \
  --output_dir /vepfs_default/chanxueyan/lhp/xh/code/CHOIR-upload/output \
  --model models/tasterob_hoi_detector.pt \
  --num_workers 1 \
  --conf 0.5
```

也支持 `--data_source taste_rob` 从 Taste-Rob 抽样；其原始数据集与
`llm_output.json` 路径仍硬编码在脚本中。`--data_source my_data` 是
`local` 的兼容别名。流水线 B 复制原视频并保留原始分辨率。

两个入口的可视化均写入：

```text
output/<VIDEO_ID>/visualizations/
├── track.mp4
├── initial_mask.jpg
└── first_frame_tracked_mask.jpg
```

### detector 伪标签与微调

发布模型 `models/tasterob_hoi_detector.pt` 可直接推理，无需重训。训练数据
不随仓库发布。`training/prepare_*.py` 中仍有旧的 Taste-Rob / SAM2 绝对路径，
复现前需改为实际路径。

```text
models/wilor_hand_detector.pt
        │
        ▼
training/prepare_data.py          # 生成 left/right 伪标签 → data/
        │
        ▼
training/train.py                 # 中间手部 detector
        │
        ▼
training/prepare_hoi_data.py      # 手部 detector + SAM2 生成 object 伪标签 → data_hoi/
        │
        ▼
training/train_hoi.py             # 三类微调 → runs_hoi/.../last.pt
        │
        ▼
models/tasterob_hoi_detector.pt   # 确认后复制为发布模型
```

在 `stage1/Yolov8` 根目录以模块方式运行：

```bash
python -m training.prepare_data
python -m training.train
python -m training.prepare_hoi_data
python -m training.train_hoi
```

可选评估：`python -m training.test`、`python -m training.test_hoi`。
`training/prepare_good_bad_cases.py` 用于整理评估用例，路径同样需按本机修改。

物体训练标签来自 SAM2 自动伪标签；`labeling.py` 只服务于流水线 A 的
人工 `first_mask.png`。

### 可移植性说明

- 推理入口默认读写仓库根目录 `data/` 与 `output/`，也可用 CLI 覆盖。
- 两个推理入口的 SAM2 权重路径仍为本机绝对路径。
- Taste-Rob 训练/抽样相关绝对路径仍硬编码。
- `sam2/_C.so` 是平台相关可选 CUDA 扩展，不纳入 Git。
- `configs/bytetrack.yaml` 相对于当前工作目录解析，请从 `stage1/Yolov8` 启动。

## Stage 1：sam-3d-objects 物体重建

### 功能

在 Yolov8 输出的 RGB 帧和物体 mask 上，用 SAM 3D Objects 重建单帧物体
mesh / pose，并可选做 silhouette 后优化。CHOIR **正式入口仅**：

```text
stage1/sam-3d-objects/run_reconstruction.py
```

默认只处理第 0 帧。

### 输入 / 输出

默认读写仓库根目录 `output/`（与 Yolov8 一致）：

```text
output/<VIDEO_ID>/
├── rgbs/0.png          # 必需
└── obj_masks/0.png     # 必需（RGBA 物体 mask）
```

主要写出到同一 `output/<VIDEO_ID>/`：

```text
glb_0.glb
intrinsics.json
transform_0.json
rendered_on_image.png
env_depths/
├── depth_0.exr
├── pc_0.ply
└── sampled_pc_0.ply
```

### 环境

CHOIR 统一使用 `hd`（路径见上文「Python 环境总览」）。请在**有 GPU 的
worker** 上运行；当前登录节点通常无 GPU，无法在此做端到端推理验证。

```bash
conda activate /vepfs_default/chanxueyan/lhp/xh/env/hd
cd /vepfs_default/chanxueyan/lhp/xh/code/CHOIR-upload/stage1/sam-3d-objects
```

`hd` 中已确认可用：`torch`、`pytorch3d`、`kaolin`、`trimesh`、`einops`。
写 `env_depths/depth_*.exr` 需要 `OpenEXR`；若 worker 上缺失：

```bash
pip install OpenEXR
```

### 模型权重

`run_reconstruction.py` 固定读取 `checkpoints/hf/pipeline.yaml`，再加载同目录下各
ckpt / yaml，以及模块根目录下的 MoGe 与 DINOv2 权重。

本机已从原仓库拷贝（**均不纳入 Git**）。在
`stage1/sam-3d-objects/` 下的文件结构：

```text
checkpoints/hf/
├── pipeline.yaml
├── ss_generator.yaml
├── ss_generator.ckpt
├── ss_decoder.yaml
├── ss_decoder.ckpt
├── ss_encoder.yaml
├── ss_encoder.safetensors
├── slat_generator.yaml
├── slat_generator.ckpt
├── slat_decoder_mesh.yaml
├── slat_decoder_mesh.ckpt
├── slat_decoder_mesh.pt
├── slat_decoder_gs.yaml
├── slat_decoder_gs.ckpt
├── slat_decoder_gs_4.yaml
└── slat_decoder_gs_4.ckpt

Ruicheng/
└── moge-v2-vitl-normal/
    └── model.pt

.cache/torch/hub/checkpoints/
└── dinov2_vitl14_reg4_pretrain.pth
```

根目录 `.gitignore` 已排除 `checkpoints/`、`Ruicheng/`、`.cache/`。

### 建议测试命令

在有 GPU 的 worker 上，激活 `hd` 并进入模块目录后直接跑：

```bash
conda activate /vepfs_default/chanxueyan/lhp/xh/env/hd
cd /vepfs_default/chanxueyan/lhp/xh/code/CHOIR-upload/stage1/sam-3d-objects

CUDA_VISIBLE_DEVICES=0 /vepfs_default/chanxueyan/lhp/xh/env/hd/bin/python -u run_reconstruction.py \
  --video_id 107407 \
  --debug
```

说明：

- `--debug`：单进程，便于排错；省略则按 GPU 数 `mp.spawn`。
- `--post_optimize`：可选，开启 silhouette 姿态 refinement（默认关闭）。
- `--video_id`：只跑指定视频；省略则扫整个 `output/`。
- 默认 `--data_dir` / `--output_dir` = 仓库根 `output/`。
- 无需额外 `export`（`LIDRA_SKIP_INIT` / `PYTHONPATH` / `TORCH_HOME` / `HF_HOME` 等）。

## Dyn_HaMR_new 环境配置

`stage1/Dyn_HaMR_new` 是例外，它不使用 `hd`，而是使用两套相互独立的
Python 环境：

- `dynhamr`：运行手部检测、跟踪和 Dyn-HaMR 优化。
- `vipe`：运行 VIPE，估计相机位姿、相机内参和深度。

当前机器上验证过的环境位于：

```text
/vepfs_default/chanxueyan/lhp/xh/env/dynhamr
/vepfs_default/chanxueyan/lhp/xh/env/vipe
```

验证过的主要版本：

| 环境 | Python | PyTorch | TorchVision | CUDA runtime |
|---|---:|---:|---:|---:|
| dynhamr | 3.10.16 | 2.0.0+cu118 | 0.15.1+cu118 | 11.8 |
| vipe | 3.10.19 | 2.7.0+cu128 | 0.22.0+cu128 | 12.8 |

以下命令均从仓库根目录开始执行：

```bash
cd CHOIR/stage1/Dyn_HaMR_new
export DYNHAMR_ROOT="$PWD"
export ENV_ROOT=/vepfs_default/chanxueyan/lhp/xh/env
```

### 1. 创建 dynhamr 环境

```bash
conda create -p "$ENV_ROOT/dynhamr" python=3.10.16 -y
conda activate "$ENV_ROOT/dynhamr"

pip install torch==2.0.0 torchvision==0.15.1 \
  --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

pip install -r requirements.txt --no-build-isolation
pip install -e .

cd third-party/DROID-SLAM
python setup.py install
cd ../hamer
pip install -e '.[all]'
pip install -v -e third-party/ViTPose
cd "$DYNHAMR_ROOT"
```

如果需要严格复现上游原始环境，也可参考
`stage1/Dyn_HaMR_new/scripts/install_conda.sh`。上游脚本使用 PyTorch
1.13.0、TorchVision 0.14.0 和 CUDA 11.7；本仓库记录的版本是当前机器实际使用的版本。

验证环境：

```bash
"$ENV_ROOT/dynhamr/bin/python" -c \
  "import torch; print(torch.__version__, torch.version.cuda)"
```

### 2. 创建 vipe 环境

```bash
cd "$DYNHAMR_ROOT/third-party/vipe"

conda env create \
  -p "$ENV_ROOT/vipe" \
  -f envs/base.yml
conda activate "$ENV_ROOT/vipe"

pip install -r envs/requirements.txt
pip install --no-build-isolation -e .
```

`envs/base.yml` 会安装 Python 3.10、CUDA 12.8 编译工具和 Eigen；
`envs/requirements.txt` 固定了 PyTorch 2.7.0+cu128 等 Python 依赖。

验证环境：

```bash
"$ENV_ROOT/vipe/bin/python" -c \
  "import torch; print(torch.__version__, torch.version.cuda)"
"$ENV_ROOT/vipe/bin/vipe" --help
```

批处理在需要时会自动调用 VIPE。默认执行 `conda activate vipe`；若环境是用绝对
prefix 创建的，请设置：

```bash
export CHOIR_VIPE_ENV=/vepfs_default/chanxueyan/lhp/xh/env/vipe
```

## Dyn-HaMR 模型下载

模型和数据应放在以下结构中：

```text
stage1/Dyn_HaMR_new/
├── _DATA/
│   ├── BMC/
│   │   ├── bone_len_max.npy
│   │   ├── bone_len_min.npy
│   │   ├── CONVEX_HULLS.npy
│   │   ├── curvatures_max.npy
│   │   ├── curvatures_min.npy
│   │   ├── joint_angles.npy
│   │   ├── PHI_max.npy
│   │   └── PHI_min.npy
│   ├── data/mano/
│   │   └── MANO_RIGHT.pkl
│   ├── droid.pth
│   ├── hamer_ckpts/
│   │   └── checkpoints/hamer.ckpt
│   ├── hmp_model/
│   │   └── results/model/
│   └── vitpose_ckpts/
│       └── vitpose+_huge/wholebody.pth
└── third-party/hamer/pretrained_models/
    └── detector.pt   # optional relative symlink → ../../../../Yolov8/models/wilor_hand_detector.pt
```

### 自动下载公开模型

先激活 `dynhamr` 环境，然后在 `stage1/Dyn_HaMR_new` 根目录执行：

```bash
conda activate "$ENV_ROOT/dynhamr"
bash scripts/prepare.sh
```

该脚本会下载：

- HaMeR 演示模型包，包括 HaMeR 和 ViTPose 权重。
- DROID-SLAM 权重 `droid.pth`。
- HMP motion-prior 模型。

手部检测器不单独下载：`launch_hamer.py` 默认使用仓库内
`stage1/Yolov8/models/wilor_hand_detector.pt`（相对路径解析）。
也可用环境变量覆盖：

```bash
export CHOIR_HAMER_YOLO_MODEL=/path/to/wilor_hand_detector.pt
```

`pretrained_models/detector.pt` 仅为可选相对符号链接；正式运行以
`--yolo_model` 传入的路径为准。

脚本中记录的其余下载地址：

```text
HaMeR:
https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT

DROID-SLAM:
https://drive.google.com/uc?id=1VD1vGhl_NPzy8mza4Fx6vvqFpnlzZ86L

HMP:
https://drive.google.com/uc?id=1LfMugcIM5WfenPkInzJGm5IEwCUK_AMy
```

### 手动准备 MANO

MANO 模型受单独许可证约束，不能随仓库重新分发。请在
[MANO 官网](https://mano.is.tue.mpg.de/) 注册并下载
`MANO_RIGHT.pkl`，放到：

```text
stage1/Dyn_HaMR_new/_DATA/data/mano/MANO_RIGHT.pkl
```

### 准备 BMC 约束数据

按照 [Hand-BMC-pytorch](https://github.com/MengHao666/Hand-BMC-pytorch)
的说明生成八个 `.npy` 文件，并放到：

```text
stage1/Dyn_HaMR_new/_DATA/BMC/
```

这些文件是优化约束数据，不应提交到 Git。

## VIPE 模型缓存

VIPE 第一次推理时会自动从 PyTorch Hub 和 Hugging Face 下载依赖模型。
为了避免模型写入用户主目录，建议固定缓存位置：

```bash
cd "$DYNHAMR_ROOT/third-party/vipe"
export TORCH_HOME="$PWD/torch_cache"
export HF_HOME="$PWD/hf_cache"
```

本地缓存通常落在：

```text
third-party/vipe/
├── checkpoints/
├── torch_cache/
└── hf_cache/
```

这些目录可能占用数 GB，不应提交到 Git。正式流水线把 VIPE 推理结果写到
`output/<VIDEO_ID>/dynhamr/vipe_results/`，而不是 `third-party/vipe/vipe_results/`。

`batch_test_videos.py` 调用 VIPE 时会自动设置：

```bash
export TORCH_HOME="$DYNHAMR_ROOT/third-party/vipe/torch_cache"
export HF_HOME="$DYNHAMR_ROOT/third-party/vipe/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

请先在本机跑通一次联网下载（或从已有机器拷贝上述缓存目录），之后即可离线复用。
不要设置 `TRANSFORMERS_CACHE` 覆盖 `HF_HOME`，否则会找不到
`hub/models--*` 布局下的权重。

显存不足时可手动：

```bash
"$ENV_ROOT/vipe/bin/vipe" infer /path/to/video.mp4 \
  --pipeline no_vda \
  --output /path/to/output/<VIDEO_ID>/dynhamr/vipe_results
```

## 运行 Dyn-HaMR

本模块在 CHOIR-upload 仓库内自包含：代码、`_DATA`、`third-party/{vipe,hamer,DROID-SLAM}`
均位于 `stage1/Dyn_HaMR_new/`；手检测权重来自同仓
`stage1/Yolov8/models/wilor_hand_detector.pt`。运行时只需外部 `dynhamr` / `vipe`
conda 环境。

数据与结果约定：

```text
output/<VIDEO_ID>/<VIDEO_ID>.mp4          # 输入（Yolov8 等上游写出）
output/<VIDEO_ID>/dynhamr/                # Hydra / 抽帧 / VIPE / HaMeR / 优化中间结果
output/<VIDEO_ID>/mano_params/            # 导出的 MANO 参数
output/<VIDEO_ID>/hand_meshes/            # 导出的手部 mesh
```

正式入口仅为 `dyn-hamr/batch_test_videos.py`（内部调用 `run_opt.py`，必要时自动跑 VIPE）。
批处理日志写在入口脚本同目录（不进 `output/<VIDEO_ID>/`）：

```text
stage1/Dyn_HaMR_new/dyn-hamr/batch_test_errors_*.txt
stage1/Dyn_HaMR_new/dyn-hamr/batch_test_results_*.txt
```

```bash
# 可选：VIPE 环境不是名为 vipe 时
# export CHOIR_VIPE_ENV=/path/to/env/vipe

conda activate "$ENV_ROOT/dynhamr"
cd "$DYNHAMR_ROOT/dyn-hamr"
python batch_test_videos.py --video_id <VIDEO_ID> --gpus 0

# 可视化
python run_vis.py --log_root ../../../output/<VIDEO_ID>/dynhamr
```

常用参数：

- `--video_id`：只跑指定视频；可传多个；省略则扫描 `output/` 下所有 `<id>/<id>.mp4`。
- `--gpus`：GPU id 列表。
- `--video_dir`：输出根目录，默认仓库根 `output/`。

## Stage 2：DexGraspNet_table 抓取训练数据

模块路径：`stage2/DexGraspNet_table/`。使用与其它非 Dyn-HaMR 模块相同的 `hd` 环境。

口径：

| 用途 | 入口 | 说明 |
|---|---|---|
| 训练抓取数据生成 | `grasp_generation/main_prep_data.py` | 桌面抓取退火优化，写出 `grasp_data/` |
| 可选后处理筛选 | `grasp_generation/validate_grasping_pose.py` | PyBullet 稳定性过滤 |
| mesh 准备（inference / 上游） | `prepare_meshdata.py` | 从 HOI mesh 打包 `meshdata/`，**不是**训练抓取入口 |

### 目录约定

```text
stage2/DexGraspNet_table/meshdata/
├── sam3d/<VIDEO_ID>/
│   ├── decomposed.obj          # 必需
│   ├── init_obj_poses.npy      # 必需（可用 scripts/generate_object_pose_mine.py 生成）
│   ├── scale.json              # 可选
│   ├── obj_points_*.ply        # 可选（prepare_meshdata FPS 写出）
│   └── grasp_data/             # main_prep_data 写出
└── dexgraspnet/<OBJECT_ID>/    # 可选的另一数据源
```

`meshdata/` 与实验日志默认在模块内，不提交到 Git。

### mesh 准备（可选，inference / 上游）

当源数据在 `output/<VIDEO_ID>/optimized_hoi_seq/`（含 `obj_canonical.obj`、`obj_00000.json`）时：

```bash
conda activate /vepfs_default/chanxueyan/lhp/xh/env/hd
cd stage2/DexGraspNet_table
python prepare_meshdata.py --video_id <VIDEO_ID>
# 默认 --source_dir 为仓库根 output/，--data_root 为本模块 meshdata/
```

若缺少 `init_obj_poses.npy`：

```bash
cd grasp_generation
python scripts/generate_object_pose_mine.py --data_root_path ../meshdata
```

### 生成抓取训练候选

```bash
conda activate /vepfs_default/chanxueyan/lhp/xh/env/hd
cd stage2/DexGraspNet_table/grasp_generation
python main_prep_data.py --object_code_list <VIDEO_ID>
# 默认 --data_root=../meshdata ，日志在 ../data/experiments/<name>/
```

依赖本地 `mano/`（含 MANO / contact 等，需按许可自行准备）以及 `manotorch`、`torchsdf`、`pytorch3d` 等。

### PyBullet 后处理筛选（可选）

```bash
cd stage2/DexGraspNet_table/grasp_generation
python validate_grasping_pose.py \
  --object_dir ../meshdata/sam3d/<VIDEO_ID> \
  --batch
```

## GraspFlowMatching（Stage 2 模型）

模块路径：`GraspFlowMatching/`。训练与推理都在此目录；使用 `hd` 环境。

| 用途 | 入口 |
|---|---|
| 训练 | `train_cam_ray.py` |
| 推理 | `sample_cam_ray_ddp.py` |
| 接触图（可选） | `compute_contact_map_per_frame.py` / `compute_contact_map_render.py` |

共享路径见 `data_layout.py`：默认 `meshdata` → `stage2/DexGraspNet_table/meshdata`，推理源 → 仓库根 `output/`，MANO → `stage2/.../mano`。

### 预训练权重

发布包内本地权重目录（已被 `.gitignore` 忽略，不会进 Git）：

```text
GraspFlowMatching/results/050-Linear-velocity-None/checkpoints/0040000.pt
```

推理请指向该 checkpoint；自行训练则会写到 `results/<exp>/checkpoints/`。

### 训练

依赖 Stage2 生成的：

```text
stage2/DexGraspNet_table/meshdata/{sam3d|dexgraspnet}/<id>/
├── decomposed.obj
├── obj_points_10000.ply
└── grasp_data/*.json
```

```bash
conda activate /vepfs_default/chanxueyan/lhp/xh/env/hd
cd GraspFlowMatching
torchrun --nproc_per_node=<N> train_cam_ray.py --results_dir results
```

### 推理

默认从 `output/<VIDEO_ID>/` 读取 HOI 序列（`optimized_hoi_seq` 或 `optimized_hoi_init_seq`），并从
`stage2/.../meshdata/sam3d/<VIDEO_ID>/` 读取规范物体 mesh/点云。可先跑本模块或 Stage2 的
`prepare_meshdata.py`。

```bash
cd GraspFlowMatching
torchrun --nproc_per_node=<N> sample_cam_ray_ddp.py ODE \
  --ckpt results/050-Linear-velocity-None/checkpoints/0040000.pt \
  --output_dir samples_ddp \
  --video_id <VIDEO_ID>
```

`sample_cam_ray_ddp.py --output_dir samples_ddp` 会写出：

```text
samples_ddp/<VIDEO_ID>/                  # 采样摘要等
samples_ddp_contact_map/<VIDEO_ID>/      # hand_*.ply / obj_*.ply（接触图输入）
samples_ddp_render/<VIDEO_ID>/<frame>/   # 渲染用分帧目录
output/<VIDEO_ID>/grasp_correction/camera_ray_depth_offset.json
```

### 接触图后处理（可选）

`--samples_dir` 须与推理时的 `--output_dir` 一致（默认 `samples_ddp`）：

```bash
# 逐帧 JSON（主 fitting 读取）
python compute_contact_map_per_frame.py --samples_dir samples_ddp --video_id <VIDEO_ID>
# -> output/<VIDEO_ID>/grasp_correction/contact_map_per_frame.json

# 写入 render 分帧目录
python compute_contact_map_render.py --samples_dir samples_ddp --video_id <VIDEO_ID>
# -> samples_ddp_render/<VIDEO_ID>/<frame>/contact_map.json
```

## 不纳入 Git 的内容

至少应排除：

```text
_DATA/
third-party/vipe/checkpoints/
third-party/vipe/torch_cache/
third-party/vipe/hf_cache/
third-party/vipe/vipe_results/
stage2/DexGraspNet_table/meshdata/
stage2/DexGraspNet_table/data/
output/
data/
**/__pycache__/
**/*.egg-info/
*.pt
*.pth
*.ckpt
*.safetensors
*.pkl
batch_test_errors_*.txt
batch_test_results_*.txt
```

请同时遵守 Dyn-HaMR、HaMeR、VIPE、MANO、DexGraspNet 和各模型文件各自的许可证。
