# AI-Lessons-for-Xiaobai

AI扫盲课 - 用大白话讲清楚人工智能的8个核心理论

## 📚 课程简介

本课程面向没有任何算法基础的小白，用大白话讲解AI的核心原理。包含**入门版**（纯大白话）和**深化版**（深入浅出）两个版本。

---

## 📁 目录结构

```
AI-Lessons-for-Xiaobai/
├── README.md                          # 本文件
├── 原始PPT.pptx                       # PowerPoint格式（入门版汇总）
├── 📁 images/                         # 可视化图表
│   ├── temperature_effect.gif         # 温度对SoftMax影响动画
│   ├── cross_entropy.gif              # 交叉熵损失变化动画
│   └── generate_gifs.py               # 图表生成脚本
│
├── 📁 入门版/                          # 大白话讲解（8课）
│   ├── 第01课-概率的世界-marp.md
│   ├── 第02课-注意力-marp.md
│   ├── 第03课-嵌入空间-marp.md
│   ├── 第04课-层层递进-marp.md
│   ├── 第05课-自监督学习-marp.md
│   ├── 第06课-涌现能力-marp.md
│   ├── 第07课-对齐理论-marp.md
│   └── 第08课-泛化与迁移-marp.md
│
├── 📁 深化版/                          # 深入浅出（8课）✨
│   ├── 第01课-概率的世界-深化版.md    # 含公式、演化史、GIF可视化
│   ├── 第02课-注意力-深化版.md        # 含Transformer架构、Notion知识库
│   ├── 第03课-嵌入空间-深化版.md      # 含CLIP对比学习
│   ├── 第04课-层层递进-深化版.md      # 含MLLM框架
│   ├── 第05课-自监督学习-深化版.md    # 含三阶段训练
│   ├── 第06课-涌现能力-深化版.md
│   ├── 第07课-对齐理论-深化版.md      # 含RLHF/PPO/GRPO/GSPO
│   └── 第08课-泛化与迁移-深化版.md    # 含LoRA/量化/KV Cache
│
├── 📄 番外篇-MLLM专题.md               # MLLM多模态大模型专题
└── 📄 Notion_MLLM_知识库提取.md        # 原始提取内容
```

---

## 🎯 两版对比

| 特性 | 入门版 | 深化版 |
|------|--------|--------|
| **目标受众** | 完全零基础 | 有基础想深入 |
| **数学公式** | ❌ 无 | ✅ 完整LaTeX |
| **论文引用** | ❌ 无 | ✅ 25+篇经典论文 |
| **可视化** | ❌ 无 | ✅ GIF动画 |
| **Notion知识库** | ❌ 无 | ✅ 已注入 |
| **页数/课** | ~8页 | ~15页 |
| **风格** | 纯大白话 | 深入浅出 |

---

## 🚀 使用方式

### 方式一：Marp幻灯片（推荐）

1. 安装 [VS Code](https://code.visualstudio.com/)
2. 安装插件 **Marp for VS Code**
3. 打开任意 `.md` 文件
4. 点击右上角预览按钮查看幻灯片
5. 导出：
   - `Ctrl+Shift+P` → `Marp: Export to PDF`
   - `Ctrl+Shift+P` → `Marp: Export to PPTX`
   - `Ctrl+Shift+P` → `Marp: Export to HTML`

### 方式二：直接阅读

所有文件均为Markdown格式，可直接阅读。

---

## 🎨 深化版特色内容

### 1. 数学可视化（GIF动画）

| 图表 | 说明 | 文件 |
|------|------|------|
| **温度对SoftMax影响** | 展示不同温度下概率分布的变化 | `images/temperature_effect.gif` |
| **交叉熵损失变化** | 展示训练过程中损失值下降 | `images/cross_entropy.gif` |

### 2. Notion MLLM知识库注入

深化版第2-8课已注入来自Notion MLLM知识库的内容：

| 课程 | 注入内容 |
|------|----------|
| 第2课 | Transformer三种架构、位置编码演进、KV Cache |
| 第3课 | CLIP对比学习、多模态嵌入、图像-文本对齐 |
| 第4课 | MLLM典型框架、分层结构、注意力数学 |
| 第5课 | 预训练两阶段、Adapter训练、对比学习损失 |
| 第7课 | RLHF完整流程、PPO/GRPO/GSPO算法对比、DPO |
| 第8课 | LoRA原理、AWQ/KIVI量化、推理优化 |

### 3. 经典论文引用

涵盖25+篇里程碑论文：
- **Vaswani et al., 2017** - Transformer
- **Devlin et al., 2018** - BERT
- **Brown et al., 2020** - GPT-3
- **Kaplan et al., 2020** - Scaling Laws
- **Ouyang et al., 2022** - RLHF/InstructGPT
- **Radford et al., 2021** - CLIP

---

## 📖 课程内容大纲

| 课次 | 主题 | 核心概念 | 深化版新增 |
|:---:|:---|:---|:---|
| 1 | 概率的世界 | 概率建模、next-token prediction | SoftMax、交叉熵、温度参数、演化史 |
| 2 | 注意力 | Self-Attention | Transformer架构、位置编码、KV Cache |
| 3 | 嵌入空间 | 向量嵌入 | 余弦相似度、CLIP、多模态嵌入 |
| 4 | 层层递进 | 分层表征学习 | MLLM框架、神经网络细节 |
| 5 | 自监督学习 | 预训练 | 三阶段训练、对比学习、MLM |
| 6 | 涌现能力 | 规模效应 | Scaling Law、Chinchilla |
| 7 | 对齐理论 | RLHF | PPO/GRPO/GSPO、DPO、奖励模型 |
| 8 | 泛化与迁移 | 迁移学习 | LoRA、量化、KV Cache优化 |

---

## 🎓 适合人群

- ✅ 对AI好奇但不懂技术的小白（入门版）
- ✅ 想系统了解AI原理的产品经理/运营（入门版）
- ✅ 有一定数学基础，想深入理解的进阶学习者（深化版）
- ✅ 需要讲授AI课程的老师/讲师（深化版）
- ✅ 准备AI相关面试的求职者（深化版）

---

## ⚙️ 导出注意事项

### 字体问题

如导出PDF/PPTX后汉字显示为方框，请：
1. 安装 **Noto Sans CJK** 字体：`brew install font-noto-sans-cjk`
2. 或导出为 **HTML** 格式（浏览器能正确渲染）

### GIF动画

Marp导出时：
- **HTML**：✅ 保留GIF动画
- **PDF/PPTX**：❌ GIF变为静态图（第一帧）

建议在讲课时用浏览器打开HTML版本展示动画。

### 公式可编辑性

Marp导出的PPTX是**栅格化**的（图片），如需编辑：
1. 使用PowerPoint 365的「图片转文字」功能
2. 或修改Markdown源文件后重新导出

---

## 📝 更新日志

### 2025-03-15
- ✅ 创建8课入门版Marp幻灯片
- ✅ 创建8课深化版（含公式、论文）
- ✅ 注入Notion MLLM知识库内容到第2-8课
- ✅ 添加GIF可视化图表（温度影响、交叉熵）
- ✅ 创建番外篇-MLLM专题
- ✅ 修复中文字体显示问题

---

## 🔗 相关资源

- [Marp官方文档](https://marp.app/)
- [LaTeX数学公式参考](https://en.wikibooks.org/wiki/LaTeX/Mathematics)
- [OpenAI GPT](https://chat.openai.com/)
- [Claude](https://claude.ai/)

---

Made with ❤️ | AI扫盲课团队

**GitHub**: https://github.com/ZhiyeTang/AI-Lessons-for-Xiaobai
