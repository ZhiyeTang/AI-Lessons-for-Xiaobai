---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: "Noto Sans SC", "Source Han Sans", "WenQuanYi Micro Hei", sans-serif;
  }
  code {
    font-family: "SF Mono", "Monaco", "Inconsolata", "Fira Code", monospace;
  }
  h1 { color: #1e3a5f; }
  h2 { color: #2c5282; }
  .map-box {
    background: #f7fafc;
    border-left: 4px solid #2c5282;
    padding: 14px 18px;
    line-height: 1.6;
  }
---

<!-- _class: lead -->

# AI 扫盲课 · 深化版

## 第2课：Tokenizer 与向量表示

> 离散文本如何进入模型，顺便补齐位置编码

---

## 课程地图

<div class="map-box">

已学：

- 第1课：概率与语言模型

本课：

- 第2课：Tokenizer 与向量表示

下节：

- 第3课：注意力与 Transformer

</div>

---

## 上节回顾

上节我们建立了一个底线：

> 模型本质上是在预测下一个 token。

但还有一个前提没讲：

**“token” 到底是什么？模型又怎么把文字变成数字？**

---

## 本节主线

今天把这条链补完整：

1. 文本先被切成 token
2. token id 再映射成向量
3. 向量之间形成语义空间
4. 位置编码告诉模型顺序信息

---

## 为什么需要 Tokenizer

如果把每个完整词语都当成一个独立单位，会遇到两个问题：

- 词汇表会爆炸
- 新词永远在出现

所以主流做法不是“按词死切”，而是：

> 用有限的子词单元，去组合无限的文本。

---

## 一个 BPE 直觉例子

### Byte Pair Encoding

从字符开始：

```text
low
lower
lowest
```

统计最常一起出现的片段，不断合并：

- `l + o -> lo`
- `lo + w -> low`
- `e + r -> er`

最后得到一种折中：

- 高频片段作为独立 token
- 低频词保留可拆分能力

---

## Tokenizer 解决了什么

它同时平衡了三件事：

- **表达能力**：常见词语不要拆得太碎
- **泛化能力**：没见过的新词也能拼出来
- **成本控制**：词表不能无限大

这也是为什么：

- 英文往往一个词对应 1 到多个 token
- 中文很多时候一个字就是一个 token 单元

---

## token id 还不够

### 模型不能直接理解整数编号

假设：

- `猫 -> 1729`
- `狗 -> 3141`

这些编号只是索引，没有语义。

真正进入模型之前，还要经过一步：

> 把 token id 查表映射成连续向量，这一步叫 Embedding。

---

## Embedding 是什么

### 把离散符号映射到连续空间

例如：

```text
"猫"   -> [0.2, -0.5, 0.8, ...]
"狗"   -> [0.1, -0.4, 0.7, ...]
"汽车" -> [-0.6, 0.9, 0.2, ...]
```

Embedding 层的价值在于：

- 让模型可以做连续计算
- 让相似概念有机会在空间里靠近

---

## 语义空间的直觉

在一个理想化的向量空间里：

- `猫` 和 `狗` 距离更近
- `猫` 和 `汽车` 距离更远

工程上常用余弦相似度衡量方向接近程度：

$$\cos(\theta) = \frac{x \cdot y}{\|x\|\|y\|}$$

所以“语义相似”通常不是字面长得像，而是向量方向更接近。

---

## 只有词义还不够

### 顺序也必须编码进去

这两句话的 token 几乎一样：

- 我打你
- 你打我

如果模型只看 token 集合、不看顺序，它会把两句话当得太像。

因此需要位置信息。

---

## Position Encoding 的作用

位置编码不是“新词义”，而是给每个 token 一个“地址”。

它解决的是：

- 谁在前
- 谁在后
- 相对距离有多远

常见表达方式之一：

$$PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$$
$$PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d})$$

扫盲课只需要记住它的目的，不必死记公式。

---

## 把整条链串起来

### 文本进入模型的流程

```text
原始文本
-> Tokenizer 切分
-> token ids
-> Embedding 向量
-> 加上位置信息
-> 送入 Transformer
```

前两课加起来，才真正解释了：

**模型在“猜下一个 token”之前，看到的到底是什么。**

---

## 本节小结

> 第2课讲的是“离散文本 -> 连续表示”的完整链条。

- Tokenizer 解决切分和词表规模问题
- Embedding 让 token 进入连续向量空间
- 语义空间让“相似概念彼此靠近”
- Position Encoding 让模型知道顺序和位置

---

## 下节预告

### 第3课：注意力与 Transformer

文字已经被变成向量之后，接下来要回答：

**模型怎么利用上下文，决定该关注哪里？**

会讲到：

- Query / Key / Value
- Self-Attention
- Multi-Head Attention
- Transformer
- KV Cache

---

<!-- _class: lead -->

## 谢谢！

**Q&A 时间**

第2课：Tokenizer 与向量表示

