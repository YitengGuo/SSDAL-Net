# SSDALNet: Synergistic Sparse-Deformable Attention Learning Network

This repository contains the official PyTorch implementation of **SSDALNet** (Synergistic Sparse-Deformable Attention Learning Network), as described in our paper:

> **Adaptive Sparse-Deformable Synergistic Mechanism for Object Detection in Complex Underwater Scenes**  
> Yiteng Guo, Ru Xu, Hao Li, Zhibin Hao, Ye Zhang  
> *School of Mechanical and Electrical Engineering, Xinxiang University, China*  

---

## 🔬 Introduction

Underwater object detection is a fundamental perceptual technology for marine exploration, ecological monitoring, and autonomous underwater vehicles (AUVs). However, existing detection algorithms suffer from severe performance degradation in complex underwater environments due to:

- **Wavelength-dependent light absorption** → feature blurring
- **Severe forward/backward optical scattering** → low contrast
- **Dynamic object deformation** → small-scale target suppression

SSDALNet addresses these challenges through an **Adaptive Sparse-Deformable Synergistic Framework** combining two novel attention mechanisms:

| Module | Location | Innovation |
|--------|----------|-----------|
| **ASSA** | Backbone (ResNet Stage3/4) | Adaptive Sparse Self-Attention with squared-ReLU sparse mask + dense attention hybrid fusion |
| **DBRA** | Neck (FPN) | Deformable Bi-level Routing Attention with region-level routing + offset-driven deformable sampling |

---

## 🏗️ Architecture

```
SSDALNet Architecture
│
├── Backbone: ResNet_SSDAL
│   ├── Stage1-2: Standard ResNet residual blocks
│   ├── Stage3:   ASSA module (squared-ReLU sparse mask + dense attention)
│   └── Stage4:   ASSA module (global semantic denoising)
│
├── Neck: DBRA_FPN
│   ├── Level-1: Region-level Semantic Routing (G×G region partition)
│   └── Level-2: Offset-driven Deformable Sampling (K=9 sample points)
│
└── Head: TOODHead (Task-aligned Object Detection)
```

---

## 📊 Main Results on DUO Dataset

| Method | AP (%) | Recall (Small) |
|--------|--------|---------------|
| SSDALNet (Ours) | **70.3** | **57.2** |
| MIP-Net | 70.1 | 54.8 |
| TOOD (baseline) | 68.4 | 51.3 |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YitengGuo/SSDAL-Net.git
cd SSDAL-Net

# Install dependencies (follows mmdetection 2.22.0)
pip install torch==1.10.1 torchvision==0.11.2
pip install mmcv==1.4.0 -f https://download.openmmlab.com/mmcv/dist/torch1.10.0/index.html
pip install -r requirements.txt
python setup.py develop
```

### Dataset Preparation

Download the DUO dataset:

```bash
# DUO dataset
git clone https://github.com/chongweiliu/DUO.git
# Organize as:
# data/udmdet/DUO/{annotations, train2017, val2017, test2017}
```

### Training

```bash
# Train SSDALNet on DUO dataset
python tools/train.py configs/ssdalnet/ssdalnet_tood_r50_fpn_2x_duo.py
```

### Testing

```bash
# Test with trained checkpoint
python tools/test.py configs/ssdalnet/ssdalnet_tood_r50_fpn_2x_duo.py \
    <path/to/checkpoint.pth> --eval bbox
```

---

## 📂 Key Modules

| File | Description |
|------|-------------|
| `mmdet/models/utils/assa.py` | ASSA: Adaptive Sparse Self-Attention with squared-ReLU |
| `mmdet/models/necks/dbra_fpn.py` | DBRA-FPN: Deformable Bi-level Routing Attention FPN |
| `mmdet/models/backbones/resnet_ssdal.py` | ResNet-SSDAL: ResNet with ASSA on Stage3/4 |
| `mmdet/models/detectors/ssdalnet.py` | SSDALNet: Complete detector integrating all components |

---

## 🔧 Configuration

Key hyperparameters for ASSA and DBRA:

```python
# ASSA in backbone
assa_key_channels=None       # Default: in_channels // 8
assa_num_heads=1
assa_temperature=1.0

# DBRA in neck
dbra_groups=4                # G×G region partition (4×4 = 16 regions)
dbra_sampling_points=9      # K deformable sample points
```

---

## 📜 License

This project is released under the [Apache-2.0 License](LICENSE).

---

## 🙏 Acknowledgement

- [mmdetection](https://github.com/open-mmlab/mmdetection) — OpenMMLab Detection Toolbox
- [DUO Dataset](https://github.com/chongweiliu/DUO) — Large-scale Underwater Object Detection Dataset

---


