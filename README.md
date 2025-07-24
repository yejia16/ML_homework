# 基于CUB-200-2011的鸟类细粒度图像分类
### —— 一项关于迁移学习与模型优化的深度探索

本项目对基于CUB-200-2011数据集的细粒度图像分类任务进行了一次全面的探索。我们通过一个系统化的方法，从一个简单的基线模型开始，逐步迭代并优化，最终实现了一个高性能的深度学习模型。

仓库中的代码展示了一系列的对比实验，包括不同模型架构（ResNet, DenseNet）、注意力机制（SE-Net）以及先进训练策略（如差分学习率、余弦退火等）的实现与分析。

## 核心功能

-   **系统性的实验设计**: 从一个简单的基线模型开始，通过一系列逻辑清晰的对比实验，逐步构建并验证一个高性能的冠军模型。
-   **先进的训练策略**: 实现了包括AdamW优化器、差分学习率、余弦退火学习率调度器在内的现代深度学习最佳实践，以进行高效的模型微调。
-   **注意力机制集成**: 将轻量且高效的Squeeze-and-Excitation (SE-Net) 注意力模块集成到ResNet架构中，以增强模型的特征辨识能力。
-   **深入的结果分析**: 提供了完整的模型评估流程，包括生成最终的性能对比表、绘制高分辨率的混淆矩阵，以及通过Grad-CAM进行模型行为的可解释性可视化。
-   **模块化与可配置的代码**: 项目核心的训练脚本高度可配置，允许用户通过命令行参数轻松地进行不同模型和超参数的组合实验。

## 数据集

本项目使用 **CUB-200-2011** 数据集。

-   **下载**: 你必须从[官方网站](http://www.vision.caltech.edu/datasets/cub_200_2011/)下载数据集。
-   **环境配置**: 下载后，解压压缩包。代码期望数据集的目录名为`CUB_200_2011`，并位于项目的根目录下。预期的目录结构如下：
    ```
    你的项目文件夹/
    ├── CUB_200_2011/
    │   ├── images/
    │   ├── attributes/
    │   ├── parts/
    │   ├── images.txt
    │   └── ...
    ├── train_baseline.py
    ├── ...
    └── README.md
    ```

## 文件结构与说明

以下是本项目中关键文件的功能说明：

-   `dataset.py`
    -   **功能**: 包含自定义的PyTorch数据集类 `CUB200Dataset`。该脚本负责读取数据集的文件结构，解析图片路径和对应标签，并应用必要的数据预处理和增强变换。

-   `train_baseline.py`
    -   **功能**: 对应**实验A**。该脚本使用预训练的ResNet-50，实现了最基础的迁移学习策略。它冻结了骨干网络的所有层，仅训练最后新添加的分类层，是整个项目的性能基准。

-   `train_improved.py`
    -   **功能**: 对应**实验B**。该脚本在基线模型的基础上，将一个SE-Net注意力模块嵌入到ResNet-50架构中。同时，它采用了更深度的微调策略，解冻了最后一个卷积阶段（`layer4`）进行训练。

-   `train_advanced.py`
    -   **功能**: 对应**实验C、D、E**，是项目中最核心、最灵活的训练脚本。它可以通过命令行参数进行高度配置，并实现了一整套高级训练技术，包括：
        -   全模型微调。
        -   支持多种模型架构 (ResNet-50, DenseNet-121, EfficientNet-B3)。
        -   使用AdamW优化器。
        -   为模型主体和分类头设置不同的学习率（差分学习率）。
        -   使用余弦退火学习率调度器。

-   `final_analysis.ipynb`
    -   **功能**: 用于所有模型训练完成后的分析与可视化工作的Jupyter Notebook。其主要功能包括：
        1.  加载所有实验中保存下来的最佳模型权重。
        2.  在测试集上评估每个模型，以生成最终的量化性能对比表。
        3.  为冠军模型生成并保存高分辨率的混淆矩阵。
        4.  生成并保存Grad-CAM可视化图，用于对比基线模型和冠军模型的注意力机制。


## 环境设置与安装

1.  **克隆本仓库:**
    ```bash
    git clone https://github.com/your_username/your_repository_name.git
    cd your_repository_name
    ```

2.  **下载数据集:** 如上所述，下载CUB-200-2011数据集并将其放在项目根目录。

3.  **创建Python虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # 在 Windows 系统中，使用 `venv\Scripts\activate`
    ```

4.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```

## 如何运行

你可以通过以下命令复现项目中的所有实验。训练好的模型权重将以`.pth`格式保存在项目根目录。

**1. 运行基线实验 (A)**
```bash
python train_baseline.py
```

**2. 运行集成SE-Net的改进实验 (B)**
```bash
python train_improved.py
```

### 3. 运行高级策略实验 (C, D, E)

使用`train_advanced.py`脚本，并通过不同的命令行参数来指定模型和超参数。

- **实验C (ResNet-50 + 高级训练策略):**
```bash
python train_advanced.py --model resnet50 --epochs 25 --lr_head 1e-3 --lr_body 1e-4
```
**实验D (DenseNet-121, 冠军模型):**
```bash
python train_advanced.py --model densenet121 --epochs 25 --lr_head 1e-3 --lr_body 1e-4
```
**实验E (EfficientNet-B3 + 高级训练策略):**
```bash
python train_advanced.py --model efficientnet_b3 --epochs 25 --lr_head 1e-3 --lr_body 1e-4
```

### 4. 进行结果分析与可视化
在所有模型都训练完毕后，使用Jupyter Lab或Jupyter Notebook打开并运行`final_analysis.ipynb`中的代码单元，即可生成最终的实验结果和所有可视化图表。
