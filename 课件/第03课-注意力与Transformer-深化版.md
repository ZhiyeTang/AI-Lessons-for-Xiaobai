---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: "Noto Sans SC", "Source Han Sans", "WenQuanYi Micro Hei", sans-serif;
  }
  h1 { color: #1e3a5f; }
  h2 { color: #2c5282; }
  .map-box {
    background: #f7fafc;
    border-left: 4px solid #2c5282;
    padding: 14px 18px;
    line-height: 1.6;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
---

<!-- _class: lead -->

# 小白AI课

## 第3课：注意力与 Transformer

> 模型如何利用上下文，而不是只会机械续写

---

## 课程地图

<div class="map-box">

已学：

- 第1课：概率与语言模型
- 第2课：Tokenizer 与向量表示

本课：

- 第3课：注意力与 Transformer

下节：

- 第4课：预训练、规模化与能力提升

</div>

---

## 上节回顾

我们已经知道：

- 文本会先被切成 token
- token 会变成向量
- 位置也会被编码进去

现在还差最关键的一步：

**模型怎么决定当前这个 token 应该看谁？**

---

## 一个典型例子

> 小明把苹果给小华，因为它很甜。

读到 `它` 时，人类会自然联想到 `苹果`。

这说明理解不是只看当前词，而是要：

- 回头看前文
- 给相关词更高权重

这就是注意力机制的直觉。

---

## Q / K / V 的大白话解释

### Query / Key / Value

可以把它理解成三句话：

- `Query`：我现在在找什么
- `Key`：我这里有哪些线索
- `Value`：根据被关注的线索，我真正提供什么内容

当 `Query` 和某个 `Key` 更匹配时，对应的 `Value` 权重就更高。

---

## Self-Attention 公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

1. `QK^T` 先算相关性分数
2. `softmax` 把分数变成权重
3. 用权重对 `V` 做加权求和

除以 $\sqrt{d_k}$ 的作用是避免分数过大，导致 softmax 过于极端。

---

## kernel 视角：attention 像数据依赖的核平滑

如果先忽略 softmax 前后的细节，attention 可以看成：

- 用 query 和 key 算相似度
- 再用相似度对 values 做加权平均

这和核方法里的 kernel smoother 很像：

$$\text{output}(q) = \sum_j \alpha_j(q) v_j$$

其中权重 $\alpha_j(q)$ 由相似度决定。

所以从研究味的角度，attention 可以被理解成：

> 一个输入依赖、内容依赖、可学习的核回归机制。

---

## Q / K / V 从哪里来

如果输入表示矩阵是：

$$X \in \mathbb{R}^{n \times d_{\text{model}}}$$

那么注意力层通常先做三次线性投影：

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

其中：

- $W_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$

所以 Q、K、V 不是“天然存在”的三样东西，而是同一份输入经过不同线性变换后的三种视角。

---

## 为什么说 attention 也是一种联想记忆

如果把：

- $K$ 看成可寻址索引
- $V$ 看成存储内容

那么 query 的作用就是用内容相似性去“读内存”。

这时 attention 的计算本质上变成：

> 用 query 在 key-value 存储中做一次软检索，再返回加权内容。

所以它同时像：

- 核回归
- 可微检索
- 联想记忆

这也是为什么 attention 在语言、视觉和多模态里都如此通用。

---

## Self-Attention 公式

<center>
<img src="../images/attention_heatmap.png" width="60%">
</center>

---

## 为什么 Multi-Head 有用

> 也就是同时使用多个 Self-Attention 进行处理

一个头只能看到一种“相关性”。

多个头可以并行学不同关系：

- 指代关系
- 语法关系
- 时间关系
- 长距离依赖

所以它不是简单重复，而是：

> 用多个视角同时看上下文。

---

## Multi-Head 的矩阵形式

第 $h$ 个头可以写成：

$$\text{head}_h = \text{Attention}(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)})$$

最后把所有头拼接起来：

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_H)W_O$$

这页最重要的结论是：

- 每个头有自己的一套投影矩阵
- 所以不同头可以学不同类型的相关性
- 最后再统一映射回模型主维度

---

## 多头的更深直觉：分块子空间里的并行核

从更研究化的角度，多头并不只是“多看几眼”。

它更像是：

- 把表示空间投影到多个子空间
- 在每个子空间里各自计算一套 attention kernel
- 再把这些局部结果合并

所以 multi-head 的价值在于：

> 允许模型在不同几何子空间里学习不同的相似性结构。

这比单头的全局单一相似度函数表达力更强。

---

## Transformer 做了什么

把注意力机制组织成一个可堆叠的大系统：

- Multi-Head Attention
- Feed Forward
- Residual / LayerNorm
- 多层堆叠

这使得模型可以逐层形成更抽象的表示。

---

## Transformer 还能怎么看：token mixer + channel mixer

很多研究会把一个 Transformer block 粗略拆成两部分：

- Attention：在 token 之间混合信息
- Feed Forward：在通道维度上做非线性变换

也就是：

- attention 负责“谁和谁交互”
- MLP 负责“每个位置内部如何重编码”

这个视角很有用，因为它解释了为什么：

> Transformer 不只是注意力堆叠，而是“跨 token 交互”和“位置内变换”交替进行。

---

## 为什么 Self-Attention 计算量高

如果序列长度是 $n$，那么相关性矩阵 $QK^T$ 的大小就是：

$$n \times n$$

这意味着核心复杂度里有一项会随序列长度近似平方增长：

$$O(n^2)$$

所以长上下文的难点不只是“记忆更多内容”，还有：

> 注意力矩阵本身会迅速变大。

这也是后来各种稀疏注意力、线性注意力和 KV Cache 优化很重要的原因。

---

## Transformer 架构图

<div class="columns">
<div>

<center>
<img src="../images/transformer_arch.png" width="68%">
</center>

</div>
<div>

常见三类用法：

- Encoder-Only：理解任务，如 BERT
- Decoder-Only：生成任务，如 GPT
- Encoder-Decoder：输入输出结构明显的任务，如翻译

目前业界，通常以 Decoder-Only LLM 为核心直觉。

</div>
</div>

---

## 为什么推理会越来越慢

如果每生成一个新 token，都把前面所有 token 重新算一遍，成本很高。

这就引出一个工程优化：

### KV Cache

- 历史 token 的 `K/V` 缓存起来
- 新 token 到来时直接复用历史结果

---

## KV Cache 直观图

<center>
<img src="../images/kv_cache.png" width="60%">
</center>

一句话理解：

> 训练强调并行，推理强调复用。

---

## 本节小结

> 第3课讲的是“模型如何利用上下文”。

- 注意力机制让模型不再只会机械按顺序扫过去
- Q / K / V 提供了“查找线索 -> 汇总信息”的计算框架
- Transformer 把注意力扩展成现代大模型的基础架构
- KV Cache 解释了推理优化的一个关键直觉

---

## 下节预告

### 第4课：预训练、规模化与能力提升

到这里为止，我们知道模型怎么“计算”了。

下一步要讲：

**它为什么越训越强、越大越能做事？**

会讲到：

- Pre-training
- Scaling Law
- Chinchilla
- In-context Learning
- Emergence

---

<!-- _class: lead -->

## 谢谢！

**Q&A 时间**

第3课：注意力与 Transformer
