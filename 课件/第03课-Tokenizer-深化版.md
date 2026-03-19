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
  .math-block { font-size: 0.9em; }
  .paper-ref {
    font-size: 0.85em;
    color: #666;
    border-left: 3px solid #2c5282;
    padding-left: 10px;
    margin: 10px 0;
  }
---

<!-- _class: lead -->

# AI 扫盲课 · 深化版

## 第3课：Tokenizer——词是如何被切分的

> 从"字"到"Token"：AI处理文本的第一步

---

## 复习回顾

### 上节课：AI = 概率计算器 + 注意力

AI说话 = 猜下一个词的游戏

**新问题**：
AI看到的"词"，和我们理解的"词"是一样的吗？

> ChatGPT看到的不是"中国"，而是["中", "国"]或["中","国"]？

---

## 一个例子

### "ChatGPT很厉害" → AI看到什么？

**人类视角**：
```
ChatGPT | 很 | 厉害
```

**AI视角（Token视角）**：
```
["Chat", "G", "PT", "很", "厉害"]
或者
["Chat", "GP", "T", "很", "厉", "害"]
```

**为什么不一样？**

---

## 为什么需要Tokenizer？

### 问题1：词汇表太大

如果每个词都是一个独立的"token"：
- 词汇表 = 无限大 💥
- AI永远学不完

### 问题2：新词不断出现

- "量子纠缠" —— 2020年前很少用
- "人工智能" —— 2016年后才火
- "绝绝子" —— 网络新词

**如何解决？** → 用子词(subword)组合！

---

## 子词(Subword)的核心思想

- **中文**：偏旁部首 → 汉字 → 词语
- **英文**：字母 → 词根 → 单词

**核心思想**：
> 用有限的基本单元，组合出无限的词汇

### 例子

| 词 | 子词拆分 |
|---|---------|
| unhappy | un + happy |
| playing | play + ing |
| 人工智能 | 人工 + 智能 |

---

## BPE算法：最主流的Tokenizer

### 全称：Byte Pair Encoding（字节对编码）

**发明者**：Sennrich et al., 2016（用于机器翻译）

**核心思想**：
1. 从最基础的字符开始
2. 统计最频繁出现的字符对
3. 合并成一个新token
4. 重复直到达到目标词汇表大小

---

## BPE算法：具体例子

### 训练数据

```
low lower lowest
new newer newest
```

### 第1步：初始化（字符级别）

```
l o w | l o w e r | l o w e s t
n e w | n e w e r | n e w e s t
```

---

## BPE算法：具体例子

### 第2步：统计频率

| 字符对 | 频率 |
|--------|------|
| e + r | 3次 |
| l + o | 3次 |
| o + w | 3次 |
| w + e | 2次 |

---

## BPE算法：具体例子

### 第3步：合并最频繁的

合并 "er"（频率最高）：
```
l o w | l o w er | l o w er s t
n e w | n e w er | n e w er s t
```

---

## BPE算法：具体例子

### 第4步：继续合并

合并 "lo"：
```
lo w | lo w er | lo w er s t
n e w | n e w er | n e w er s t
```

合并 "low"：
```
low | low er | low er s t
n e w | n e w er | n e w er s t
```

---

## BPE训练后的词汇表

### 最终词汇表（示例）

```
基础字符：a, b, c, ..., z
合并子词：er, lo, low, ne, new, est, ...
```

**优势**：
- 低频词 = 基础字符组合（如"量子" = 量 + 子）
- 高频词 = 直接作为一个token（如"low"）
- 未知词 = 总能拆成字符

---

## 不同语言的Token数量

| 内容 | Token数量 |
|------|----------|
| "hello" | 1 token |
| "人工智能" | 2-3 tokens |
| "ChatGPT很厉害" | 6-8 tokens |

**有趣现象**：
- 英文通常1个词 = 1-2 tokens
- 中文通常1个字 = 1-1.5 tokens
- 这就是为什么中文AI处理更"贵"

---

## Tokenizer对AI的影响

### 影响1：成本计算

AI按token收费：
- 输入token数 + 输出token数 = 总费用

**省钱技巧**：
- 中文 → 尽量用简洁表达
- 英文 → 相对更省token

---

## Tokenizer对AI的影响

### 影响2：代码处理

```python
# 这段代码有多少token？
def hello_world():
    print("Hello, World!")
```

**答案**：约15-20 tokens

### 影响3：上下文长度

GPT-4的128K上下文 = 约10万汉字
（因为中文每个字约1.3 tokens）

---

## 不同类型的Tokenizer

| 类型 | 代表 | 特点 |
|------|------|------|
| **BPE** | GPT系列 | 合并频繁字符对 |
| **WordPiece** | BERT | 基于概率最大化 |
| **SentencePiece** | T5, LLaMA | 语言无关，统一处理 |
| **Unigram** | ALBERT | 从大到小删减 |

**趋势**：
- 大模型时代 → 词汇表越来越大（10万+）
- 多语言支持 → 统一Tokenizer

---

## 本节小结

> ### Tokenizer = AI的"分词器"> 
003e - **核心任务**：把文本切分成AI能处理的token
003e - **主流算法**：BPE（合并频繁字符对）
003e - **关键概念**：子词(subword)平衡词汇表大小和表达能力
003e - **实际影响**：成本、长度限制、多语言处理

---

## 关键问题

**为什么"量子纠缠"可能被拆成["量","子","纠","缠"]？**

因为：
1. 训练数据里这个词出现频率不够高
2. 没达到合并成单独token的阈值
3. 但["量","子"]各自可能已经是token

**这说明什么？**
> AI的"理解"受限于训练数据的频率！

---

## 下节预告

### 第4课：提示词工程——如何让AI听懂你的话

💡 **核心问题**：
同样的意思，换个说法，AI的回答可能天差地别。

**将学习**：
- 什么是好的Prompt
- Few-shot学习技巧
- 思维链(CoT)方法

---

<!-- _class: lead -->

## 谢谢！

**Q&A 时间**

第3课：Tokenizer 🔤
