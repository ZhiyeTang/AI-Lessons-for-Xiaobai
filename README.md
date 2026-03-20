# AI扫盲课 - 10课深化版

用大白话讲清楚大模型、后训练、多模态与图像生成的 10 节课件。

## 课程大纲

| 课次 | 主题 | 核心内容 |
|:---:|:---|:---|
| **1** | 概率与语言模型 | Next-token prediction、Softmax、交叉熵、温度 |
| **2** | Tokenizer 与向量表示 | BPE、Embedding、语义空间、位置编码 |
| **3** | 注意力与 Transformer | QKV、Self-Attention、Multi-Head、KV Cache |
| **4** | 预训练、规模化与能力提升 | Pre-training、Scaling Law、Chinchilla、ICL、Emergence |
| **5** | 提示词工程与上下文学习 | Role/Context/Task、Few-shot、CoT、Prompt 边界 |
| **6** | 幻觉、检索与工具使用 | Hallucination、RAG、Function Calling、评测与校验 |
| **7** | 指令微调与模型“变听话” | SFT/IFT、任务数据格式、LoRA、多模态 SFT |
| **8** | 偏好学习与强化学习后训练 | Preference Data、RLHF、PPO、DPO、GRPO、GSPO |
| **9** | 多模态理解基础 | CLIP、表示对齐、视觉编码器、VQA/OCR/文档理解 |
| **10** | 图像生成基础 | Diffusion、Latent Diffusion、Flow Matching |

## 课程逻辑

```text
第1-3课：模型如何接收输入、表示信息、利用上下文
第4课：能力从哪里来
第5-6课：人如何更好地使用模型、约束模型
第7-8课：模型如何被后训练成更会做事、更符合偏好
第9-10课：能力如何扩展到图像理解与图像生成
```

## 本次重编排重点

- 按《课程重编排建议》从 8 课升级为 10 课版。
- 补齐 `Tokenizer -> Embedding -> Position Encoding -> Attention` 的知识链。
- 将原“对齐理论”拆成两课：
  - 第7课：SFT / IFT / LoRA
  - 第8课：Preference Learning / RLHF / DPO / GRPO / GSPO
- 将“多模态理解”与“图像生成”拆开，避免概念混讲。
- 统一每课的 `课程地图 / 上节回顾 / 本节小结 / 下节预告` 结构。

## 使用方式

1. 安装 VS Code 与 Marp for VS Code。
2. 打开 `课件/` 下任意 `.md` 课件。
3. 使用 Marp 预览。
4. 按需导出 PDF / HTML / PPTX。

## 目录结构

```text
课件/
├── 第01课-概率与语言模型-深化版.md
├── 第02课-Tokenizer与向量表示-深化版.md
├── 第03课-注意力与Transformer-深化版.md
├── 第04课-预训练、规模化与能力提升-深化版.md
├── 第05课-提示词工程与上下文学习-深化版.md
├── 第06课-幻觉、检索与工具使用-深化版.md
├── 第07课-指令微调与模型变听话-深化版.md
├── 第08课-偏好学习与强化学习后训练-深化版.md
├── 第09课-多模态理解基础-深化版.md
└── 第10课-图像生成基础-深化版.md

课件_old/
└── 旧版 8 课与早期深化版备份

images/
├── attention_heatmap.png
├── cross_entropy.gif
├── kv_cache.png
├── mllm_arch.png
├── temperature_effect.gif
└── transformer_arch.png
```

## 更新记录

### 2026-03-20

- 升级为 10 课深化版目录。
- 统一所有课件的课程衔接页和术语口径。
- 重构第 7、8 课，拆分 `SFT` 与 `偏好学习 / RL 后训练`。
- 新增第 10 课《图像生成基础》。
- 清理旧版 8 课总结页中遗留的编号与完结文案。

## 仓库

https://github.com/ZhiyeTang/AI-Lessons-for-Xiaobai

