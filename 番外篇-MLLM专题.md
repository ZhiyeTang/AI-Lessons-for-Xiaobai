---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  }
  h1 {
    color: #1e3a5f;
  }
  h2 {
    color: #2c5282;
  }
---

<!-- _class: lead -->

# AI 扫盲课 · 番外篇

## MLLM（多模态大语言模型）专题

> 基于Notion MLLM知识库整理

---

## 什么是MLLM？

**MLLM** = Multimodal Large Language Model

**核心能力**：
- 不仅能理解文字
- 还能理解图像、视频、音频
- 实现跨模态的理解与生成

**代表模型**：GPT-4V、Claude 3、Gemini、Qwen-VL、GLM-4V

---

## MLLM vs LLM

| 特性 | LLM | MLLM |
|------|-----|------|
| 输入 | 纯文本 | 文本+图像/视频/音频 |
| 输出 | 纯文本 | 文本+可引用图像内容 |
| 应用场景 | 聊天、写作 | 图像理解、视频分析、OCR |

**本质区别**：
- LLM：只能"读"文字
- MLLM：能"看"图+"读"字+"理解"关系

---

## 第一部分：Transformer回顾与扩展

### 三种Transformer架构

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Encoder-Only   │  Decoder-Only   │  Encoder-Decoder│
│    (BERT)       │    (GPT)        │    (T5/BART)    │
├─────────────────┼─────────────────┼─────────────────┤
│ 双向注意力      │ 单向注意力      │ 编码器+解码器   │
│ 适合：理解任务  │ 适合：生成任务  │ 适合：翻译/摘要 │
│ 分类、NER       │ 文本生成        │ 跨模态生成      │
└─────────────────┴─────────────────┴─────────────────┘
```

**MLLM通常采用**：Decoder-Only或Encoder-Decoder架构

---

## 位置编码在MLLM中的演进

### 传统位置编码

**绝对位置编码**：
```python
# Sinusoidal位置编码
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### MLLM中的创新

**2D位置编码**（用于图像）：
- 图像不是1D序列，需要(x,y)坐标
- 每个图像patch有两个位置编码

**3D-RoPE**（GLM-4.1V）：
- 引入时间维度，处理视频
- 位置 = (x, y, t)

---

## 第二部分：MLLM训练流程

### 三阶段训练

```
阶段1: 预训练 (Pre-training)
    ↓ 大规模图文数据
阶段2: 指令微调 (IFT/SFT)
    ↓ 指令-回答对
阶段3: 对齐训练 (RLHF/DPO)
    ↓ 人类偏好数据
    最终模型
```

---

## 阶段1：预训练 - CLIP对比学习

### 核心思想

让**图像特征**和**文本特征**在向量空间中靠近

```python
# 1. 分别提取特征
I_f = image_encoder(I)    # [n, d_i]
T_f = text_encoder(T)     # [n, d_t]

# 2. 投射到相同维度并归一化
I_e = l2_normalize(I_f @ W_i)
T_e = l2_normalize(T_f @ W_t)

# 3. 计算相似度矩阵
logits = I_e @ T_e.T * exp(t)

# 4. 对称对比学习损失
loss = (cross_entropy(logits, labels) + 
        cross_entropy(logits.T, labels)) / 2
```

**关键**：对角线元素是正样本，其他是负样本

---

## 阶段2：指令微调 (SFT)

### 什么是SFT？

**Supervised Fine-Tuning**：用(指令, 回答)对继续训练

**数据格式**：
```json
{
  "instruction": "描述这张图片",
  "input": "[image]",
  "output": "图中是一只橘色的猫坐在窗台上..."
}
```

**目标**：让模型学会按指令回答问题

---

## 阶段3：对齐训练 (RLHF)

### RLHF回顾

**Reinforcement Learning from Human Feedback**

```
模型生成回答 → 人类排序 → 训练奖励模型 → PPO优化
```

### MLLM中的变体

| 算法 | 特点 | 代表模型 |
|------|------|----------|
| **PPO** | 需要价值模型，训练稳定 | InstructGPT |
| **GRPO** | 组优势估计，无需VM | DeepSeek-R1 |
| **GSPO** | 序列级重要性权重 | Qwen3-VL |

---

## 第三部分：高效微调技术

## LoRA：低秩适应

### 核心思想

**不微调全部参数**，只训练少量"适配器"

```python
# 原始权重: W (大矩阵)
# LoRA分解: W' = W + A × B
# A: [d, r], B: [r, d], r << d

# 训练时只更新A和B
# r通常为8-64，远小于原始维度
```

**优势**：
- ✅ 显存占用少（只需存A、B）
- ✅ 训练速度快
- ✅ 可合并回原始模型

**初始化技巧**：
- A：高斯初始化
- B：零初始化（保证训练初期不改变原模型）

---

## 第四部分：推理优化

## KV Cache详解

### 问题

自回归生成时，每步都要重新计算所有token的K、V

**计算量 = O(n²)，随着序列增长，越来越慢**

### 解决方案

**缓存之前的Key和Value**

```python
# 第t步推理时
Q_t = W_q @ x_t          # 新token的Query
K_t = W_k @ x_t          # 新token的Key
V_t = W_v @ x_t          # 新token的Value

# 使用缓存
K = concat(K_cache, K_t)  # [1, t+1, d]
V = concat(V_cache, V_t)  # [1, t+1, d]

# 只计算新的注意力
attention = softmax(Q_t @ K.T / sqrt(d))
output = attention @ V

# 更新缓存
K_cache = K
V_cache = V
```

**效果**：从O(n²)降到O(n)，推理速度大幅提升

---

## KV Cache量化 (KIVI)

### 问题

KV Cache占用的显存 = batch_size × seq_len × hidden_dim × 2 × 2 bytes

长序列时，**显存爆炸** 💥

### KIVI方案

**观察**：
- Key Cache：存在异常值，需要column-wise量化
- Value Cache：分布平滑，可以正常量化

**方案**：
- Key：分组的column-wise量化 + FP16 buffer
- Value：正常per-channel量化

**效果**：
- 2-bit量化 → 显存减少75%
- 几乎不损失精度

---

## AWQ量化

### 核心观点

**权重不同等重要！**

- 小部分"显著权重"对结果影响巨大
- 对这些权重保持高精度
- 其他权重可以低精度

**公式**：
```
W_q = round(W × s) / s
```

其中s是缩放因子，根据激活值分布确定

---

## 第五部分：MLLM模型对比

## 主流MLLM对比

| 模型 | 发布时间 | 核心亮点 | 适用场景 |
|------|----------|----------|----------|
| **GLM-4.1V/4.5V** | 2025-07 | AIMv2-Huge, 3D-RoPE | 视频理解 |
| **Qwen2.5-VL** | 2025-01 | 动态分辨率, mRoPE | 文档/图像 |
| **MiniCPM-V 4.5** | 2025-08 | 3D-Resampler | 端侧部署 |
| **Qwen3-VL** | 2025-09 | MRoPE-Interleave, GSPO | 多模态推理 |

---

## Qwen2.5-VL 关键技术

### 动态分辨率

**问题**：图像大小不一，怎么处理？

**方案**：
- 将图像切分为固定大小的patch
- 根据图像尺寸动态调整patch数量
- 每个patch有2D位置编码

### mRoPE

**Multimodal Rotary Position Embedding**

- 文本：1D位置
- 图像：2D位置 (x, y)
- 视频：3D位置 (x, y, t)

### Window Attention

- 图像局部使用窗口注意力
- 减少计算量
- 保持长程依赖

---

## 总结

### MLLM核心技术栈

```
基础架构: Transformer + 位置编码扩展
    ↓
预训练: CLIP对比学习
    ↓
微调: LoRA高效适配
    ↓
对齐: RLHF/GRPO/GSPO
    ↓
推理: KV Cache + 量化
    ↓
部署: AWQ/KIVI压缩
```

---

## 与AI扫盲课的关联

| 本课内容 | 对应AI扫盲课 |
|----------|-------------|
| Transformer架构 | 第2课(注意力) + 第4课(分层) |
| CLIP预训练 | 第5课(自监督学习) |
| RLHF对齐 | 第7课(对齐理论) |
| LoRA微调 | 第8课(泛化与迁移) |
| KV Cache优化 | 第2课(注意力机制的实际应用) |

---

<!-- _class: lead -->

## 谢谢！

### MLLM专题 · 番外篇

**Q&A 时间**

---

## 参考资料

- Notion MLLM知识库
- CLIP: Learning Transferable Visual Models
- LoRA: Low-Rank Adaptation of Large Language Models
- KIVI: A Tuning-Free Asymmetric 2bit Quantization
- AWQ: Activation-aware Weight Quantization
- Qwen-VL Technical Report
- GLM-4V Technical Report
