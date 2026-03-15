# AI扫盲课 · 深化版

AI扫盲课的升级版，从"大白话"到"深入浅出"。

## 📚 课程升级说明

本深化版在保留原有"大白话"讲解的基础上，新增：

### 1. 数学原理（每课1-2页）
- LaTeX/MathJax格式的核心公式
- Softmax、注意力计算、词向量相似度等
- 损失函数、优化目标详解

### 2. 经典论文（每课1页）
- 里程碑论文引用与解读
- 架构图与演进时间线
- 作者、会议、影响力说明

### 3. 优化课程结构
```
封面（1页）
引入/问题（1页）
大白话解释（2-3页）← 保留原有内容
数学原理/公式（1-2页）← 新增 ⭐
经典论文/架构图（1页）← 新增 ⭐
应用实例（1页）
小结（1页）
```

## 📖 课程内容

| 课次 | 主题 | 新增数学内容 | 新增论文 |
|:---:|:---|:---|:---|
| 第1课 | 概率的世界 | Softmax、交叉熵、温度参数 | GPT系列、Bengio 2003 |
| 第2课 | 注意力 | Scaled Dot-Product Attention、Multi-Head | Transformer (Vaswani et al., 2017) |
| 第3课 | 嵌入空间 | 余弦相似度、位置编码 | Word2Vec (Mikolov et al., 2013)、BERT |
| 第4课 | 层层递进 | 神经网络、激活函数、层归一化 | AlexNet、ResNet |
| 第5课 | 自监督学习 | MLM、自回归损失、对比学习 | BERT、GPT系列 |
| 第6课 | 涌现能力 | Scaling Law、幂律公式 | Kaplan et al., 2020、Chinchilla |
| 第7课 | 对齐理论 | RLHF三阶段、PPO、KL散度 | InstructGPT (Ouyang et al., 2022) |
| 第8课 | 泛化与迁移 | 偏差-方差分解、迁移学习公式 | Pan & Yang 2010、MAML |

## 🎯 使用方式

### 方式一：Marp幻灯片（推荐）

1. 安装 [VS Code](https://code.visualstudio.com/)
2. 安装插件 **Marp for VS Code**
3. 打开任意 `第XX课-XXX-深化版.md` 文件
4. 点击右上角预览按钮，即可查看幻灯片
5. 可导出为 PDF / PPTX / HTML

### 方式二：直接阅读

所有文件均为Markdown格式，可直接阅读。

## 📁 文件结构

```
AI扫盲课-Marp-深化版/
├── README.md
├── 第01课-概率的世界-深化版.md
├── 第02课-注意力-深化版.md
├── 第03课-嵌入空间-深化版.md
├── 第04课-层层递进-深化版.md
├── 第05课-自监督学习-深化版.md
├── 第06课-涌现能力-深化版.md
├── 第07课-对齐理论-深化版.md
└── 第08课-泛化与迁移-深化版.md
```

## 📚 参考论文汇总

### 基础架构
- **Attention Is All You Need** (Vaswani et al., 2017) - Transformer
- **BERT** (Devlin et al., 2018) - 双向预训练
- **GPT-3** (Brown et al., 2020) - 涌现能力

### 规模定律
- **Scaling Laws** (Kaplan et al., 2020) - OpenAI
- **Chinchilla** (Hoffmann et al., 2022) - DeepMind

### 对齐
- **RLHF** (Ouyang et al., 2022) - InstructGPT
- **Constitutional AI** (Bai et al., 2022) - Anthropic

### 词向量与预训练
- **Word2Vec** (Mikolov et al., 2013) - Google
- **GloVe** (Pennington et al., 2014) - Stanford
- **AlexNet** (Krizhevsky et al., 2012) - 深度学习黎明
- **ResNet** (He et al., 2016) - 残差连接

### 迁移学习
- **Transfer Learning Survey** (Pan & Yang, 2010)
- **MAML** (Finn et al., 2017)

## 🎓 适合人群

- ✅ 有一定数学基础，想深入理解AI原理的学习者
- ✅ 需要讲授AI课程的老师/讲师
- ✅ 想从"会用"进阶到"懂原理"的AI从业者
- ✅ 准备AI相关面试的求职者

## 📊 与原版对比

| 维度 | 原版 | 深化版 |
|------|------|--------|
| 目标受众 | 零基础小白 | 有数学基础的进阶学习者 |
| 数学公式 | 无 | 有（LaTeX） |
| 论文引用 | 无 | 详细 |
| 架构图 | 无 | 有 |
| 页数/课 | ~8页 | ~12页 |
| 风格 | 纯大白话 | 深入浅出 |

## 🔗 相关资源

- [原版课程](../README.md)
- [Marp官方文档](https://marp.app/)
- [LaTeX数学公式参考](https://en.wikibooks.org/wiki/LaTeX/Mathematics)

---

Made with ❤️ | AI扫盲课 · 深化版
