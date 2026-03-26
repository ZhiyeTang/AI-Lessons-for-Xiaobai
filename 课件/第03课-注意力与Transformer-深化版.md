---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&display=swap');
  section {
    font-family: "Noto Sans SC", "Noto Sans CJK SC", "Source Han Sans SC", "Source Han Sans CN", "Microsoft YaHei", "PingFang SC", "Heiti SC", "SimHei", sans-serif;
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

## 第3课：Transformer 机制与 KV Cache

> 从多头注意力、Block 结构到自回归推理优化

---

## 课程地图

<div class="map-box">

已学：

- 第1课：概率与语言模型
- 第2课：从文本到 Transformer 输入

本课：

- 第3课：Transformer 机制与 KV Cache

下节：

- 第4课：预训练、规模化与能力提升

</div>

---

## 上节回顾

上节我们已经把输入送进了 Transformer：

- 文本先变成 token ids
- token ids 变成表示矩阵 $X$
- 位置矩阵 $P$ 加进去，得到 $\tilde X = X + P$
- 然后 $\tilde X$ 生成 $Q/K/V$，做第一次 attention

这节不再问“输入从哪里来”，而是问：

**一个 Transformer block 内部到底怎样工作，为什么推理还能被工程化优化？**

---

## 本节主线

1. Transformer 的其他组成部分
2. multi-head 为什么比单头更强
3. 残差连接
4. KV Cache 为什么能避免重复计算

---

## 一个 Transformer block 里到底有什么

<div class="columns">
<div>

<center>
  <img src="../images/transformer_arch.png" width="68%">
</center>

</div>
<div>

把注意力机制组织成一个可堆叠的大系统：

- Multi-Head Attention
- Feed Forward
- 残差连接
- 多层堆叠

这使得模型可以逐层形成更抽象的表示。

</div>
</div>

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

## 残差连接为什么重要

【闪客】深入解读 Kimi 爆火论文，马斯克都转了！到底什么是注意力残差？白话解读哟
https://www.bilibili.com/video/BV1MMw1zaESW/?share_source=copy_web&vd_source=4b7c9295aa3879aa80e96dcfa95562c3&t=55

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

> 第3课讲的是“Transformer block 怎样工作，以及推理怎样被优化”。

- attention 可以看成一次可微的联想检索
- multi-head 让模型在多个子空间里并行建模相关性
- Transformer block 是 token 交互和位置内重编码的交替堆叠
- causal mask 把 block 变成了自回归生成器
- KV Cache 解释了为什么推理时可以复用历史计算

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

第3课：Transformer 机制与 KV Cache
