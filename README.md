# 音频生成模型评估方法复现

本项目汇集了多种用于音频生成领域的评估指标的实现代码，旨在为研究人员和开发者提供一个方便、统一的评估工具集。

## 目录

- [简介](#简介)
- [评估指标与模型列表](#评估指标与模型列表)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [使用说明](#使用说明)
- [致谢](#致谢)

## 简介

随着音频生成技术的快速发展，如何客观、准确地评估生成音频的质量与对齐度变得至关重要。本项目收集并复现了当前领域内多种主流的评估方法与相关的生成模型。每个子目录都对应一个独立的工具或模型，方便用户根据需求进行配置和使用。

## 评估指标与模型列表

本项目目前包含以下评估方法与模型：

*   **AudioLDM**: 一个强大的文本到音频生成模型。此目录包含其训练、微调、推理和评估的完整代码。
*   **CLAPScore**: 基于 CLAP (Contrastive Language-Audio Pretraining) 模型的评估指标，用于计算音频和文本描述之间的语义相似度分数。
*   **CLIPScore**: 基于 CLIP (Contrastive Language-Image Pretraining) 模型的评估指标。在音频领域，它通常通过计算音频频谱图和文本描述之间的相似度来工作。
*   **Fréchet Audio Distance (FAD)**: 一种广泛使用的评估生成音频质量的无参考指标。它通过比较生成样本和真实样本在预训练模型（如VGGish）嵌入空间中的分布来计算距离。
*   **LPIPS (Learned Perceptual Image Patch Similarity)**: 一种衡量两张图像相似度的感知损失。在音频任务中，可用于评估生成音频的频谱图与参考频谱图的相似度。
*   **ReWaS (Read, Watch and Scream!)**: 一个新颖的从文本和视频生成音频的模型。它以视频作为条件控制，引导文本到音频的生成过程。其代码库中也包含多种评估脚本，如能量均方根误差(Energy MAE)、CLAP Score、音视频对齐分数(AV-align Score)等。

## 项目结构

```
.
├── AudioLDM-training-finetuning-main/  # AudioLDM 模型的训练、微调和评估
├── CLAPScore_for_LASS-main/          # CLAPScore 计算
├── CLIPScore-main/                   # CLIPScore 计算
├── FAD_fadtk-main/                   # Fréchet Audio Distance (FAD) 计算工具
├── LPIPS/                            # LPIPS 感知相似度计算
├── ReWaS/                            # ReWaS 模型实现与相关评估脚本
└── README.md                         # 本文档
```

每个子目录都是一个独立的评估方法或模型实现，包含了其自身的代码、依赖项和使用说明。

## 环境配置

由于每个工具都是独立的项目，它们的依赖项各不相同。**强烈建议为每个子项目创建独立的Python虚拟环境**，以避免依赖冲突。

请进入相应的子目录，并根据其内部的 `requirements.txt` 或 `pyproject.toml` 等文件来安装所需依赖。

**示例 1：使用 pip 安装 (以 `CLIPScore-main` 为例)**
```bash
cd CLIPScore-main/
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

**示例 2：使用 poetry 安装 (以 `FAD_fadtk-main` 为例)**
```bash
cd FAD_fadtk-main/
pip install poetry
poetry install
```

## 使用说明

请优先参考每个子目录下的 `README.md` 文件（如果存在）以获取最详细的使用指南。以下是简要说明：

### AudioLDM
该目录包含了 AudioLDM 模型的完整流程。
- **训练**: 参照 `bash_train.sh` 脚本。
- **评估**: 参照 `bash_eval.sh` 脚本。
- **推理**: 运行 `audioldm_train/infer.py`。

### CLAPScore
- **主程序**: `CLAPScore_for_LASS-main/main.py`。

### CLIPScore
- **主程序**: `CLIPScore-main/clipscore.py`。
- **使用示例**: 参考 `example/` 和 `flickr8k_example/` 目录下的脚本。

### Fréchet Audio Distance (FAD)
`FAD_fadtk-main` 提供了一个名为 `fadtk` 的便捷命令行工具。
- **命令行使用**: `python -m fadtk --gt-path /path/to/real_audio --test-path /path/to/generated_audio`。
- 更多用法请参考其内部文档。

### LPIPS
此目录下的脚本可用于计算两个图像或两个图像文件夹之间的LPIPS分数，可用于比较声谱图。
- **比较两张图片**: `lpips_2imgs.py`
- **比较两个文件夹**: `lpips_2dirs.py`

### ReWaS
- **生成音频**: 运行 `test.py`，并提供视频和文本输入。
- **评估**: `evaluation/` 目录包含了多个评估脚本：
  - `eval_MAE.py`: 计算能量均方根误差。
  - `clap_score.py`: 计算 CLAP 分数。
  - `av_align_score.py`: 计算音视频对齐分数。

## 致谢

本项目是对以下优秀开源项目的汇集与整理。所有版权归原作者所有。
*   **AudioLDM**: [https://github.com/haoheliu/AudioLDM](https://github.com/haoheliu/AudioLDM)
*   **CLAP**: [https://github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)
*   **CLIPScore**: [https://github.com/jmhessel/clipscore](https://github.com/jmhessel/clipscore)
*   **fadtk (FAD)**: [https://github.com/google-research/fadtk](https://github.com/google-research/fadtk)
*   **LPIPS**: [https://github.com/richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
*   **ReWaS**: [https://github.com/naver-ai/rewas](https://github.com/naver-ai/rewas)

感谢这些项目作者的杰出工作与开源贡献。




