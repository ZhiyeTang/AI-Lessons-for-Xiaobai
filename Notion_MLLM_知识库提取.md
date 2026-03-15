# MLLM知识库提取内容

> 来源: Notion MLLM知识库
> 提取时间: 2026-03-15

---


# Transformers

## 核心

      ### Transformer典型结构
      Transformer模型是一个标准的Encoder-Decoder结构。
      - Encoder每个层包含一个自注意力模块和一个前馈传播模块；对任意输入序列，会使用一个位置编码叠加到输入信号，便于模型感知到输入序列的顺序。
      - Decoder每个层包含一个掩码自注意力模块、一个跨注意力模块和一个前馈传播模块。掩码自注意力模块用于屏蔽输出序列中的未来内容。Decoder也可以接收其他输入来生成目标序列，因而可以融合不同模态的特征。
      Transfoer的Encoder和Decoder可以单独构建模型使用，也可以组合使用： 
      - Encoder-Only模型：适用于需要理解输入的任务，如句子分类和命名实体识别。 
      - Decoder-Only 模型：适用于生成任务，如文本生成。
      - Encoder-Decoder模型（Seq2Seq模型）：适用于需要根据输入进行生成的任务，如翻译或摘要。 

  ## 注意力机制
  典型的注意力机制是关于序列、和的函数：
  其中，三个序列的尺寸分别定义为[B, N, D]，使用点积评价每个Query对所有Key的关联程度，即每个Query在所有Key上的注意力分布。
  的作用主要目的是防止点积的值过大。当嵌入的维度太大时，点积会出现异常大的值。后果有以下两个：
  1. 点积异常大的值可能导致softmax计算的指数项异常大，导致出现非常集中的分布；使用这个除法可以将点积的值限制在一个固定的平滑规模。
  1. 防止出现梯度爆炸或梯度消失。

## 位置编码

## 注意力模型

# MLLM大模型

> 📊 数据库: MLLM大模型对比

# 常见框架和技术

## 训练

  ### CLIP

    ```python
    # 分别提取图像特征和文本特征
I_f= image_encoder(I)#[n, d_i]
T_f= text_encoder(T)#[n, d_t]

# 对两个特征进行线性投射，得到相同维度的特征，并进行l2归一化
I_e= l2_normalize(np.dot(I_f, W_i), axis=1)
T_e= l2_normalize(np.dot(T_f, W_t), axis=1)

# 计算缩放的余弦相似度：[n, n]
logits= np.dot(I_e, T_e.T)* np.exp(t) # t: 温度

# 对称的对比学习损失：等价于N个类别的cross_entropy_loss
labels= np.arange(n)
# 对角线元素的labels
loss_i= cross_entropy_loss(logits, labels, axis=0)
loss_t= cross_entropy_loss(logits, labels, axis=1)
loss= (loss_i+ loss_t)/2
    ```

## 微调

  ### LoRA
    > 大模型微调到底有没有技术含量，或者说技术含量到底有多大？ - 猛猿的回答 - 知乎
    在微调的时候，相比起直接微调完整的，LoRA认为可以将W分解成两个奇异矩阵和的乘积。这有两个好处：1. 当的尺寸相当大时，和仍然可以是比较小的尺寸，因而降低计算复杂度；2. 在这种情况下，的乘积注定满足，而MLLM的特定领域能力往往也是只存在于少数几个秩当中，因此这样的建模能够在强化模型的特定能力的同时，不损害模型的通用能力。
    在训练时，一般对进行高斯初始化，进行0初始化。这样能够保证训练刚开始的时候模型仍然能具有预训练后的良好性能。
    由于同时也表示了LoRA时的秩；一般来说可以设置为一个比较大的值（多次实验下的最大值）。在调参的过程中，一般会使用一个比较大的，保证LoRA能够学习到新的知识，此时设置。在后续实验中，不断尝试更小的，以保证LoRA训练出来的知识秩更低、信息更精炼；而固定的此时正好能保证新知识对模型的影响能力（幅值），以获得比较良好的训练效果。

## 量化

  ### AWQ
    AWQ（Activation-aware Weight Quantization）量化是一种基于激活值分布（Activation Distribution）挑选显著权重（Salient Weight）进行量化的方法，其不依赖于任何反向传播或重建，因此可以很好地保持LLM在不同领域和模式上的泛化能力，而不会过拟合到校准集，属训练后量化（Post-Training Quantization，PTQ）的一种。
    - 核心观点1：权重并不同等重要，仅有小部分显著权重对推理结果影响较大
    - 核心观点2：量化时对显著权重进行放大可以降低量化误差
    参考链接：

  ### KIVI
    Problem: Along with the sequence gets longer, the KV cache also grows, which hinders the efficient computation of GPU cores. —> IO time dominates inference due to large KV cache size.
    Solution: KV cache quantization.
    Challenge: Outliers ruin the quantization.
    Observation: Key cache appears column-wise —> Column quantization; Value cache appears smooth —> Normal quantization.
    Introducing a FP16 buffer for grouping column-wise quantization.

## 推理

  ### KV Cache
    在自回归式生成过程中，当推理第个token时，只需要输入第个token即可，前面的注意力矩阵都可以复用。
    如上图所示，每一步的计算其实是需要根据新给的Query token计算注意力矩阵，和过往的Query token是完全无关的。所以我们需要把先前每一步的和缓存起来，而不需要对进行任何缓存，所以叫KV Cache。

### CLIP

# 常见指标

## BLEU

## CIDEr

## METEOR

## ROUGE

# MLLM基本训练过程
MLLM的典型框架通常包含一个模态特征提取器（Encoder）、模态对齐投影器（Adapter）、LLM推理核心。

## 预训练
  通常，这一阶段主要训练的是Adapter，训练数据是图像-文本的数据对，目的是保证Adapter能够将Encoder提取的模态特征投影到LLM能够识别的token空间。

## 指令微调（IFT/SFT）
  IFT阶段的目的是为了让模型能够完成用户的不同任务，因而称为“指令微调”。IFT的一个核心是要提高模型的零样本泛化（Zero Shot）能力，即能够处理没见过的任务的能力。通常，使用的数据是一个指令及对应的输入-输出对，指令以自然语言的形式给出，某些时候甚至为空字符串。
  这一阶段通常同时微调Adapter和LLM。

## 对齐微调（AFT/RLHF）
  这个阶段是为了进一步提升模型的回答质量，满足人类的预期和偏好。一般地，这个阶段会使用RLHF进行训练。
  AFT的数据一般是对模型响应的反馈（满意程度）。RLHF包含量个关键阶段：
  - 奖励模型训练：收集大量的，训练一个奖励模型，最小化以下损失函数：
  - 强化学习：使用PPO，根据RM返回的奖励分数，提升模型的回答质量。
  此外，DPO训练策略也可以用于替换RLHF实现人类偏好对齐，其损失函数定义为：
  要注意的是，由于MLLM的序列预测特性，先前输出的也会作为模型的上下文输入给模型，得到下一个step的每个token的分布。因此，DPO成为了可能。
  具体理解分析请参见 。

# 强化学习算法
在MLLM的语境下，强化学习算法一般特指PPO和GRPO算法。由于LLM时代的来临，Q-Learning、DQN乃至传统AC这样的框架都已经被扫进了故纸堆里了，实在是引人唏嘘。

## PPO
  在PPO训练的框架下，和其他AC框架算法一样，需要同时训练策略模型（PM）和价值模型（VM）。PM负责决策，即输出一个per-token distribution决定每个token的输出概率；VM负责根据决策估计当前状态的价值。总地来讲，随着训练过程的推进，可以预期PM能够做出质量更高的决策，而VM能够进行更准确的价值估计。优化目标如下所示：
  其中优势函数通过GAE构造（具体分析详见 ）：
  可以简单介绍一下这条式子：
  - 首先，表示的是一个优势函数，表示在当前时刻下，输出获得的“好处”，即预期的未来收益相比预期的当前状态的收益高了多少。一般来说，是一个基于VM定义的标量，通常使用GAE（广义优势估计）进行估计。
    - 的比值则是一个重要性采样权重。RLHF训练使用的数据通常是RefM收集的，在训练PM时为了保证监督信号没有偏差（unbiased），需要估计出当前决策如果是PM做的可以达到什么优势。
      > 在传统RL语境下，奖励函数通常是“决策（token）级”的。然而，在LLM/MLLM中，输出的单个token难以构造具体的意义以获取奖励信号，所以实际上奖励是“序列级”的。而PPO此处的重要性参数也是一个决策级的重要性权重，却用来修正一个序列级的信号，这显然不准确。随着Qwen3系列模型提出的 算法实现了这部分修正。
  - clip操作的目的是防止PM输出的分布出现非常剧烈的分布。当PM的分布和RefM的分布出现非常大的差异时，将这个比值钳制在一个比较小的范围内，保证训练稳定。

## GRPO
  GRPO摆脱了对VM的依赖，其核心是不使用基于VM定义的优势函数来指导优化的方向，而是直接使用RM来指导优化的方向：
  简单来讲：
  - 组优势函数的定义是对所有输出的奖励进行组归一化，得到一个服从标准正态分布的奖励组，每个奖励值表示该动作相比起当前状态的标准收益（收益均值，即我们以前说的价值基线）好多少。回忆在传统的PPO当中，这个价值差异是使用VM来估计的，而GRPO直接使用全组收益定义差异，是一种更节省训练资源（不需要VM）的方案。
    > GRPO存在一个本质缺陷。在PPO当中，由于价值基线本身是由VM估计的，我们可以视为这个估计方式是无偏的（价值基线=PM的收益期望）。但是在GRPO的体系中，这个价值基线是简单地对组取平均得到的，而实际上每个输出的决策概率并不相等。一个无偏的价值基线估计应该表示为：
      GRPO的构造中，这个价值基线并不完全是组奖励的平均，而是使用了奖励的标准差进行放缩。这一定程度上缓解了价值基线的无偏性，但理论上仍存在改进空间。

## GSPO
  回忆一下GRPO的优化目标：
  可以看到，这里使用的重要性权重和 中的别无二致。前文在讨论PPO时，我有提到：
  而GSPO就是沿着这个方向做的改进。我们可以思考，怎么从一个决策级的重要性权重拓展到一个序列级的重要性权重。设回答，那么很显然，模型输出一个完整序列的概率是
  那么显然，更新的重要性权重应该是这样一个形式：
  这个建模存在一个问题，就是模型实际上决策的单位还是token。所以，我们还需要将这个序列级的权重转换回决策级。GSPO的做法是，对这个权重进行几何平均：
  即假设每个决策对重要性权重的贡献完全一致，对权重开次方。实际实现采取了下式的形式，但和上式是完全等价的：

# 其他小知识

## 广义优势估计GAE
  我们知道，GAE的公式可以被很简单地整理成如下形式：
  但这个形式其实是时序差分（、）家族的一系列整理。我争取在这一块给大家一次讲透。
  首先我们先来聊一个问题：为什么需要做时序差分？我们知道，强化学习的核心其实是对决策树的一种高效搜索。而在现代深度强化学习的语境里，由于状态空间、动作空间的高度膨胀，以及我们难以对每个任务的状态转移方程进行显式建模，这种搜索往往需要以间接的方式来做：通过每一个动作的收益期望来决定当前状态下的最优决策。
  > 所以，我们面对的其实是这样一个问题：
    在状态下，模型会生成一个策略，表示每个动作的选择倾向（概率）。选择不同的动作会转移进不同的，这样就展开了一棵决策树。我们想做的无非就是在状态下，根据不同的数学手段去估计不同动作背后隐藏的决策子树的收益期望而已。
  聪明的你一定能想到：假如我沿着一条动作路径，一直搜索到底，把得到的奖励累加起来。在模型持续训练的过程当中，它会按照自己的策略（喜好）去搜索这棵决策树，这就能够保证它选择每一条动作路径的概率是服从自己本身策略分布的，也就能保证收益期望是准确（无偏，unbiased）的。恭喜你，你发明了最基础的蒙特卡洛估计（Monte-Carlo Estimation，后文简称MC估计）：
  > 这是我们展开的第一条式子，我会稍微讲细一些。后续的式子大致都以一个方式展开，就不过多赘述一些在这里设定过的内容。
    表示时刻下后续序列（设长度为）的总收益。是一个折扣因子，这在强化学习当中很常见，主要作用有两个：1. 防止模型过分注重未来收益，而忽视了近期收益；2. 在一些无限循环任务中，由于我们没有办法搜索到底，只能使用折扣因子保证不会无限叠加。
  MC估计这么简单、基础，那一定会有很大的问题，事实上确实如此。MC方法一个“臭名昭著”的问题就是方差大。显而易见，MC估计的值和序列的长度是相关的，但很多时候并不是序列越长收益就一定越高。而且，MC估计依赖于对完整序列的观测，也就是只有在决策完一条完整的序列后才能得到收益。有没有办法能够在序列中间就得到收益呢？
  我们可以从序列末尾开始看这个问题：对于终止状态，它的收益就一定是；而对于它之前的某个状态，其收益也一定是。也就是说，假如我们能够保证后续序列每一个状态的收益都估计准确了，我们就一定能以递归的方式估计出当前任意状态的收益：
  但是，在时刻下，模型还没有经历后续序列，我们怎么知道后续序列的收益呢？聪明的你一定能想到：用一个神经网络来近似这个收益不就好了？这个模型的训练就使用前后两个状态的收益来监督：
  那么，每个状态的收益显然就可以通过这个模型来逼近：
  恭喜你，你发明了单步时序差分（Temporal Difference）估计，而这个模型的监督信号我们也可以称为单步的时序差分误差（Temporal Difference Error，TD Error）。
  > 实际上，在最开始的强化学习当中，是以表格的形式存在的，也就是为每个状态维护一个数值。强化学习先驱们通过这样的精巧建模将决策树搜索问题巧妙转化成“查表”问题，就可以使用贪心的方法简单得到最优决策序列了，也就是早期价值迭代的思想。
    后来，随着任务的复杂化，这个表格也逐渐演变成了我们今天常见的神经网络。在“正统”的强化学习理论当中，神经网络也被称之为价值模型（Value Model，VM），用来估计每个状态的价值。我们也在此简单带过一下价值的定义：在模型的特定策略下，当前状态的期望总收益。
  我们也可以针对上述的VM做一个简单的变形。假如我们不止使用单步的信号进行监督，也对收益的逼近进行一个多步的推广，我们就可以很简单地得到步时序差分估计：
  VM的监督信号可以通过对单步TD Error进行简单拓展得到，此处不做展开。值得注意的是，当，此时的定义就和MC估计是完全一致的了。
  从到步TD，我们可以注意到，这些估计都是预设的，也就是说，当你决定使用哪种方法构造收益之后，在完整训练当中就只能使用这种方式。这带来的问题是，模型的时间视野是固定的，只能看到你设定的步长的未来收益。聪明的你一定会猜测，是否存在这样一种方法，能同时融合不同长度的TD估计呢？那我们就直接把不同长度的TD估计加起来，再对它们取平均：
  其中，表示收益的步长。但是这种方式存在一个显著问题：序列长度会显著影响收益的值。所以我们需要使用一个和有关的权重来控制不同步长的收益对的贡献，并且这个权重的总和必须为。我们采取指数的建模，引入一个，我们惊讶地发现，这个权重满足性质：
  所以，我们就可以建立起这样一条收益式子：
  恭喜你，你发明了。它本质上是对不同步长的收益的一种加权平均，并统一了到MC估计的完整框架。当时，收敛为；当时，收敛为MC估计。
  我们前面提到的，从MC到，本质上都是在估计状态的价值。但事实上，我们往往关注的是每个状态下，执行不同动作会带来的增量收益。我们可以引入一个新的概念，叫动作价值函数（Action Value Function），通常使用符号来表示。既然我们有了价值函数，我们也就可以构建一个差值，表示不同动作带来的增量收益：
  恭喜你，你发明了优势函数。从直觉上看，一个动作带来的增量收益越大，优势就越明显；而一个越坏的动作往往可能带来很小的增量收益，甚至是负的收益值。
  假如我们使用TD作为动作价值函数，并引入的思想对优势函数进行多步的平均，那么这个优势函数就可以写成：
  我们可以在式子中引入几个辅助项，就可以将项展开成：
  注意到，引入的项是可以互相抵消的，因此上式可以被进一步整理成：
  为了方便起见，我们直接用一个变量来定义
  然后我们要提出的这个优势函数就可以被写成
  注意到上式中，内层的求和是一个与无关的过程，因此可以直接对括号内的部分使用等比级数求和得到
  恭喜你，你发明了广义优势估计（General Advantage Estimation，GAE）。

> 📄 子页面: VideoMobileCLIP大作战
  > 本文始撰于2026年春节前夕。

  # Paper List

    ## 专题：MobileCLIP的里程碑 - Milestone of MobileCLIP

    ## 专题：数据即是正义 - Data High
      1. [ICLR 2025] Revisit Large-Scale Image–Caption Data in Pre-training
      1. [NeurIPS 2024] ShareGPT4Video: Improving Video Understanding and Generation with Better Captions
      1. [arXiv 2025] Low-hallucination Synthetic Captions for Large-Scale Vision-Language Model Pre-training
      1. [ICML 2024] VideoPrism: A Foundational Visual Encoder for Video Understanding

    ## 专题：提速之旅 - Journey to the Speed
      1. [ECCV 2024] VideoMamba: State Space Model for Efficient Video Understanding
      1. [ICLR 2023] Token Merging: Your ViT But Faster
      1. [ECCV 2024] An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models
      1. [NeurIPS 2025] FastVID: Dynamic Density Pruning for Fast Video Large Language Models
      1. [ACL 2025 Findings] PruneVid: Visual Token Pruning for Efficient Video Large Language Models

    ## 专题：从CLIP到MLLM - From CLIP to MLLM
      1. [arXiv 2023] MobileVLM: Vision Language Model for Mobile Devices
        [arXiv 2024] MobileVLM V2: Faster and Stronger Baseline for Vision Language Models
      1. [CVPR 2024] LLaVA-NeXT-Video: Stronger Video Understanding with AnyRes
      1. [arXiv 2025] A Survey of MultiModal Large Language Models
      1. [TMLR 2025] Long Context Transfer from Language to Vision

    ## 专题：黑魔法 - Black Magics
      1. [CVPR 2024] VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking
      1. [ICLR 2024/2025] V-JEPA: Video Joint Embedding Predictive Architecture
      1. [CVPR 2024] InternVideo2: Scaling Video Foundation Models
      1. [ICLR 2024] LanguageBind: Extending Video-Language Pretraining to N-modality

  # MobileCLIP
    MobileCLIP是端侧化CLIP领域的里程碑。Apple通过MobileCLIP，实现了高效的端侧视觉语义理解。我个人认为，MobileCLIP的提升来源很重要的一点是端侧训练的RIRO（Rubbish-In-Rubbish-Out）特性，一个更富有语义信息的、更低噪声的数据集必然能够带来更强的性能。Apple的DataCompDR数据集，是使用原始的DataComp-1B数据集清洗增强得到的，并且加入了一组个更强大的Teacher CLIP的知识。具体而言，对DataComp的原始图文对，对图像进行多种增强得到Teacher CLIP组生成的图像编码；使用一个较大的VLM（Caption Generator）得到一组合成的文本描述，同样也输入到Teacher CLIP组当中得到一组文本编码。
    至于网络结构上，主要是提出了一个Text-RepMixer，是由FastViT中提出的RepMixer拓展得到的，主要用于高效快速提取文本编码，并使用对卷积模块重参数化的技巧，将训练时的Conv-BN块在测试时数学等价地合并成一个独立的卷积层，从而提高端侧推理速度。这部分公式推导见 。
    最后讲一下损失函数。损失函数主要由标准的CLIP损失和知识蒸馏损失组成。其中蒸馏损失由I2T和T2I两个部分组成，但是是互相对称的，因此只从其中之一展开即可。
    其中表示两个嵌入矩阵之间的相似度：
    表征每个图像在个文本描述上的相似度分布。所以实际上，这个蒸馏损失就是将个名师天团的双模态嵌入蒸馏到MobileCLIP上，从而通过个Teacher CLIP的共同知识进一步压制MobileCLIP学到噪声信息的概率。
    对于不熟悉原始CLIP的读者，这里也一并展开的数学解析式：
    其中有
    是经典的InfoNCE损失的表达式。

  # MobileViCLIP

  # SigLIP & SigLIP 2

    ## SigLIP
      SigLIP是很经典的一篇文章，Google为了在自研的TPU上高效训练，设计了一个Sigmoid-based损失函数，在小显存-小批量训练条件下，依然训练出非常强大的视觉-文本模型。SigLIP的核心思想就是这样的一个损失函数（见下图）：
      对于标准的CLIP损失，所有的个图像-文本对的嵌入数据均需要放置在一个设备上，并构造出大小的相似度矩阵。这是由于InfoNCE损失的SoftMax式计算本质需要计算一个全局分布，这个全局分布必须保证所有样本均放置在同一个设备上才能完成计算；相比之下，Sigmoid函数是一个点对点计算，每个设备只需要管理局部样本；在所有局部样本计算完成后，再组合成最后的损失函数参与反向传播。其数学表达式为：
      其中是一个符号矩阵，在对角线上为+1，其余值为-1，表示样本的匹配性；b是一个-10初始化的可学习的参数矩阵，用于防止负样本占主导导致的冷启动训练塌陷。

    ## SigLIP 2
      相比较于SigLIP仅在损失函数上进行的改进，SigLIP 2对训练配方进行了大刀阔斧的创新。SigLIP 2的训练是两阶段的：
          1. 除了使用SigLIP的Sigmoid-based损失函数以外，此阶段还将SigLIP 2输出的（未池化）图像嵌入送入一个标准的Transformer（即右图的AR Decoder），通过进行三个任务的稠密预测实现LocCa训练：
            1. 图像描述：通过提取的图像特征描述图像；
            1. 稠密描述：给定边界框坐标，描述框内区域的内容；
            1. 指代表达：给定图像中特定区域的文本描述，输出对应的边界框坐标。
            这里的数据除了任务a以外，其余均为合成数据，通过n-grams分词和OVOD（Open Vocabulary Object Detection）合成出区域文本和边框坐标。
            此外，这里还使用了一个并行预测的技巧，有50%的概率会将输入的下三角矩阵变成一个全掩码矩阵，强制模型在没有任何token记忆的情况下直接预测完整的文本caption。
          1. 在训练的后20%进程中，SigLIP 2还使用自蒸馏（SILC）和掩码预测（TIPS）的方式进一步提升模型性能。
            1. SILC通过使用视觉编码器的EMA权重作为Teacher模型，接收完整图像视图输入，并产生一个对完整图像的语义嵌入；而Student模型则只接收完整视图裁剪所得的多个局部视图，并通过简单的辅助MLP头进行局部特征融合，与完整图像的语义进行匹配；
            1. TIPS的做法是对Teacher和Student都提供相同的完整图像输入，但Student的输入中有50%的图像块会被掩码覆盖。
            此外，为了防止新追加的蒸馏损失破坏图文对对齐知识，这两路数据流是完全独立的，即：在构造蒸馏损失时，仅使用局部视图与掩码视图；而图文对齐任务的损失仍通过数据集的原始输入得到。
      除此以外，SigLIP 2还有两个关键改进：
      - NaFlex：通过将输入图像等比例放大到Max Token Len所能承载的上限，并在图像末尾追加0填充，同时使用一个掩码记录真实图像像素位置，保证SigLIP 2可以接收任意比例图像输入；并且，由于原始的位置编码也是在正方形输入上构造的，SigLIP 2将位置编码插值到对应的比例，从而保证模型也能感知到不同图像切块在原始图像中的真实位置与比例。
      - 主动数据筛选（Active Data Curation）：这其实是一个小模型蒸馏手段。在预训练得到一个较大尺寸的SigLIP 2模型之后，在1B量级的数据上微调出数据判别能力；在蒸馏到小模型上时，使用大模型对“超批次（Super-batch）”进行打分，实际训练仅使用超批次中前一定百分比的数据进行训练，从而提升小模型的蒸馏效率。

  # MVBench
    MVBench是一篇聚焦在视频理解基准搭建的文章。
    - 任务定义：通过将9个静态图像QA的任务进行动态迁移，构造了20个要求模型具备良好动态视频理解能力的任务。
    - 数据标注：使用ChatGPT在11个公开视频基准数据集上，根据这20个定义的任务构造问题，然后提取现有的标注构造回答模板；对于开放性问题（简答题），则使用ChatGPT生成一组错误选项，并统一所有答案的长度，然后通过计算选项正确率算出得分。
    - 这20个任务分别包括：
      - 动作 (Action)：包含动作序列检索、动作预测、动作反义词判别、细粒度动作识别和意外动作检测 。
      - 物体 (Object)：涵盖特定事件中的物体存在性判定、交互物体识别，以及物体位置洗牌追踪 。
      - 位置 (Position) & 场景 (Scene)：要求确定物体的运动轨迹方向，以及特定动作发生的时间段定位 。场景任务涉及判断视频中场景如何过渡切换 。
      - 计数 (Count)：包含计算特定动作的执行次数，以及执行特定动作的物体数量 。
      - 属性 (Attribute)：要求判定特定给定时刻运动物体的外观特征，以及判定物体状态在视频周期内是否发生变化 。
      - 姿态 (Pose) & 字符 (Character)：要求从相似选项中识别细粒度姿态类别 。字符任务要求判定视频中字母出现的特定顺序 。
      - 认知 (Cognition)：包含基于当前导航指令预测后续动作的自我中心导航、对剧集片段中人物和事件的推理，以及反事实推断 。

  # Revisit Large-Scale Image-Caption Data in Pretraining Multimodal Foundation Models
    来自Apple的这篇论文是为了探索一个问题：怎么样更好地合成数据、怎么样更好地用合成数据给对视觉语言模型（包括但不限于CLIP、MLLM和扩散模型）进行训练。
    他们主要做了三件事：
    1. 使用两阶段微调构造一个MLLM Image Captioner：
      1. 描述驱动的微调：使用100万条高质量人类标注数据（Stage-1-1M）结合OCR提取的文本对MLLM进行微调，使其能够严格遵循指令输出特定长度的文本
      1. 人类对齐的微调：引入高密度描述数据集（Stage-2-HA），进一步微调模型以生成复杂且低幻觉的场景描述
    1. 使用MLLM Image Captioner输出四种形式的标注数据：
      - Short Synthetic Captions（SSC）：简短的合成文本描述
      - Descriptive Synthetic Captions （DSC）：描述性合成文本描述，使用不超过78 tokens长度的文本描述图像，包含中心主体和关键视觉元素
      - Dense Synthetic Captions + （DSC+）：超过100 tokens的高密度描述，覆盖背景和复杂关系
      - AltText Fusion Captions（AFC）：使用数据集中原始的AltText与合成的DSC融合得到
    1. 探索CLIP、MLLM和Diffusion这三种模型架构对四种数据形式的偏好：
      - 基于对比学习架构的CLIP相比起DSC与DSC+这类长文本描述，更偏好SSC这类简短的描述；并且，AltText对CLIP理解图像语义信息有不可替代的作用。高质量的真实AltText对CLIP的语义理解能力有跨越式的提升。
        - 实际上，CLIP依然能从DSC中提取深层语义信息，指标上骤降的原因主要是由于DSC引入了和评估基准不匹配的文本长度和结构分布导致的。事实上，通过 能够证明，在DSC上预训练的CLIP有着不亚于SSC预训练版本的性能。
        - CLIP作为基础视觉模型，需要覆盖极为广泛的视觉概念；而MLLM重写的合成描述通常存在模式塌陷的问题，词汇的多样性远低于原始的AltText，因此仍然需要AltText维持特征空间的多样性。
      - MLLM（以MM1为代表）更偏好长文本描述，尤其是DSC+这种具备高密度复杂信息的文本描述。
        - 尽管Image Captioner在生成DSC+时，由于强制要求Image Captioner输出高密度视觉信息，导致DSC+本身不可避免地存在更多幻觉噪声，但MLLM对这种噪声具备天然的抗性。相反，密集信息带来的上下文逻辑构建的增益，远大于局部的幻觉所带来的负面影响。
      - Diffusion模型（以SD3为代表）则更偏好DSC这种中长文本描述，能够显著提升Diffusion对文本指令的遵循能力。
        - 为什么是DSC不是DSC+？对于CLIP和Diffusion模型，由于其Text Encoder的长度被限制在77 tokens，所以无法在CLIP和Diffusion上使用DSC+进行实验。

  # ShareGPT4Video
        这篇文章主要有三个核心成果：
        1. ShareGPT4Video数据集，包含40K个由GPT-4V标注的视频-描述对，集成了如Panda-70M、Ego4D、BDD100K等开放基准数据集中的视频。
          这里使用了滑动窗口差异（DiffSW）策略对视频进行精准标注。对每一条视频，首先要求GPT-4V标注其首个关键帧，然后将首个关键帧及其标注描述与第二个关键帧一同送入GPT-4V，要求模型标注两帧之间的差异（差分描述）；最后完成逐帧标注后，通过GPT-4总结出一段连贯、具备精确时间动态和详尽空间细节的完整视频描述。
        1. ShareCaptioner-Video模型，支持四种工作模式的视频描述生成模型，并被用于标注了4.8M个高质量视频数据。
        1. ShareGPT4Video-8B模型：使用LLaVA-Next-8B初始化的MLLM。

  # VideoPrism
    尽管这篇文章的核心贡献是提出了一个强大的视频理解模型VideoPrism，但放在 之下是有原因的。在讨论VideoPrism的同时，作者们也很好地回答了一个问题：如何通过特定的训练策略设置，来高效利用海量、多源、异构的有噪数据？
    VideoPrism的出发点是一个巨量的混合质量语料库，包含3600万条高质量的人工标注视频-描述对与5.82亿条有噪视频-描述对。这些有噪数据是采用包括但不限于ASR转录、视频元数据与MLLM生成等方式构造的。在经过严格去重与测试集防泄漏之后，VideoPrism经历了两阶段的训练策略：
    1. 第一阶段采用标准的视频-文本对比学习，实现VideoPrism视频编码器与文本编码器的语义对齐。这一步的训练还引入了交替梯度下降（Alternating Gradient Descent，AGD）。在多源数据集当中，由于单个数据集内部的距离接近、而数据集间的距离较远（low intra-dataset distance, high inter-dataset distance），导致对比学习在这种训练设置下是非常难奏效的。如果对多源数据集中的数据进行随机抽取，容易出现很多“简单负样本”，因此数据集B的样本一定与数据集A的样本极不相似。AGD的思想就是在一个mini-batch当中只使用一个数据集的数据，从而保证batch内部的数据具有充分的难度，保证模型学习到更细粒度的深层次特征差异。
    1. 作者们发现，在第一阶段的对比学习当中，由于文本中的噪声，模型学到的知识偏向于外观描述，而缺少了运动相关的理解信息。因此，需要构造二阶段的掩码视频建模训练，构建了一个同参数但冻结的Video Encoder作为Teacher模型，与VideoPrism进行蒸馏训练。此时，Teacher模型接收完整的、无掩码的视频输入，而VideoPrism接收掩码视频作为输入，输出的视频嵌入会被送入两个辅助Decoder当中，分别实现全局的信息对齐与Token粒度的信息对齐。
      - Decoder 1只接收未遮蔽区域的嵌入，同时不添加任何位置编码，需要VideoPrism的Video Encoder能只通过可见区域的无序碎片尽可能精准地表达完整视频画面的信息；这一步主要是防止VideoPrism过分关注运动信息，导致原本的全局视频理解能力退化。
      - Decoder 2接收全部区域的乱序嵌入，并添加错乱的位置编码，从而要求Video Encoder需要理解视频块之间的运动关系，才能为下游的Decoder 2提供丰富的视频块位置信息，帮助Decoder 2能够将视频块路由到合适的位置上，最小化损失。
        这个设计本质上是为了抵抗二阶段训练可能出现的“捷径学习”现象，防止Decoder 2简单地将原始嵌入直接输出，以陷入局部最优解。
        我在 还增加了一些个人的理解，希望对各位有帮助。

  # ToMe
    ToMe这篇文章从头到尾只做一件事：提速。提速的思想很简单：现代的视觉感知网络都是采用的ViT作为基础设计单元，只要能以更少的token（更短的序列）进行计算，那么就一定能够实现更快的推理。
    那怎么减少序列当中的token数量呢？作者们注意到，在Attention模块输出的token序列当中，其实存在相当多的冗余信息，这些冗余信息事实上是可以被融合，可以通过一个更短的token序列完整表达原始的信号。于是，他们提出了使用Attention模块当中的Key-Embedding之间的相似度作为评价指标。我们知道，Attention模块的核心计算式是
    其中、和分别表示三个不同的token序列。作为查询，从序列中提取一组“键”，再从中查找对应的“值”。如果在计算当前时，序列的两个token的相似度比较高，那这两个token事实上是可以被直接融合成1个token，从而降低下一层的计算消耗的。
    所以，他们设计了一个二分软匹配的聚类方法，通过将序列中的token交错地均分成两个集合（Set  v.s. Set ），为Set 中的每个token找到它在Set 中最相似（余弦相似度最高）的一个匹配，得到个token对的相似度，然后将相似度最高的个对进行池化融合，从而在每次ToMe操作时可以固定减少个token，最终使得序列的长度越来越短，计算效率也随之提升。
    这个思想的巧妙之处在于，相比起其他基于裁剪（Pruning）的方案，需要手动管理被裁减的token的梯度反向传播问题，池化融合的方案由于引入的是可微算子，并不会导致被融合的token无法计算梯度，因此梯度流的传播更自然。
    当然，由于token的丢失，会导致最后求取的SoftMax分布是有偏的。融合的含义其实是使用一个token替换原来的多个token，而如果不对这个求分布进行任何干预，那相当于被合并的token上的注意力被丢失了。因此，需要额外引入一个向量，表示每个token所融合的token数量。最后计算SoftMax时，对于一个融合了个token的token，其注意力矩阵的值为：
    可以注意到，此时相当于给SoftMax函数的自变量进行了换元：
    这种基于token融合的策略还会带来一个收益：在图像上，相似的物体部件会被自然地归并在同一个token中，形成类似部件分割（Part Segmentation）的效果 。在多帧视频中，它甚至能实现对同一个物体（或部件）在整个运动轨迹上的追踪合并。

  # An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Model
    相比较于 ，本文提出的FastV是一个完全服务于推理过程的插件。作者们发现，在MLLM推理的过程当中，图像部分token占据的注意力远小于其他文本/系统提示词对应token的注意力。因此，他们认为，在推理阶段，对图像块token进行一定的删减，是可行的推理加速方案。
    他们提出的做法是：从第层开始，计算每个图像token受到的注意力分数：

    ```python
    # Q.shape = [B, N, D]
# K.shape = [B, N, D]
Attn = torch.softmax(
    Q @ K.transpose(-2, -1) / math.sqrt(D), dim=-1)  # [B, N, N]
key_attn = Attn.mean(dim=-2)                         # [B, N]
    ```
    然后直接过滤掉注意力最低的比例的图像token，从而显著提高推理速度。

  # MobileVLM & MobileVLM 2
    这里主要关注MobileVLM和MobileVLM 2中所使用的Projector：LDP和LDPv2的技术细节。在VLM当中，Vision Encoder提取的视觉嵌入需要通过一个Projector（或Adapter）进行“转码”，才能送入到LLM基模当中。
    上图展示了MobileVLM（左）和MobileVLM 2（右）的大致架构。Pointwise Conv表示一个卷积核大小为1的卷积操作。

  # Appendix

    ## MobileCLIP的卷积重参数化详解
      Coming soon…

    ## Conditional Position Encoding（CPE）
      这里简单介绍一下CPE的核心思想。关于位置编码的前世今生，我们已经在 进行了非常深入的讨论。实际上，CPE的原理远没有当前在MLLM领域广受好评的 （RoPE）复杂，本质是通过一个DW-Conv生成的一组位置编码，也就是可学习的位置编码的一种表现形式。区别在于，传统的位置编码接收到的都是一组形状的嵌入列表，而CPE首先将这个形状的嵌入列表还原（reshape）成一个形状的三维嵌入张量，然后使用2D DW-Conv的输出作为每个嵌入的位置编码，从而通过卷积的方式提供了既有绝对位置又有相对位置的编码信息。

    ## Linear Probing（线性探测实验）
      即：在预训练好的Backbone后接一个非常简单的单层线性网络，实验当中冻结Backbone、仅训练分类头。若Backbone在线性探测实验中有良好的表现，可以说明它能够提取包含极强信息量和区分度的语义信息。

    ## VideoPrism的Decoder 2
      二阶段训练的Decoder 2其实设计的很巧妙，不得不佩服Google大牛们的天才想法。我认为，这个设计主要有以下两个目的：
      1. 切断捷径学习：在掩码建模中，如果按原序输入，Decoder 2很容易学会一条极其偷懒的捷径：把未掩码（unmasked）的tokens直接“复制粘贴”到输出端 。为了打破这种捷径，模型在将token序列喂给Decoder 2之前进行了随机打乱 。并且，Decoder 2是在token被打乱之后，才给它们加上位置编码的 。
      1. Decoder 2倒逼：Decoder 2需要做精确路由（Routing）；但关键在于，它又被刻意设计得非常“弱”，这就带来：
        1. 因为Decoder 2很浅，它自身没有能力去进行复杂的时空推理。
        1. 它要想完成重建任务，就极其依赖VideoPrism传过来的特征。
        1. 这就要求VideoPrism提取的不仅仅是静态的“这是一只猫的腿”的特征，而必须是高度浓缩了时空动态的特征，比如“这是一只猫的腿，且它正处于向上跳跃动作的中间帧阶段”。只有把运动状态（时序关系）深深烙印在embedding里，浅层的Decoder 2才能将其与对应的位置编码匹配上，完成拼图。

    ## Mamba Out：从SSM到S6再到Vision Mamba
      当我们谈论人工智能领域近几年的架构进展时，绕不开的一个里程碑就是Mamba架构。其实我更愿意将Mamba视为RNN架构的一个延伸。

      ### S4: Structured State Space for Sequences
        State Space Model的核心思想是与RNN一脉相承的：输出应当视为输入与一个隐藏状态的函数：
        其中，、和均是可学习参数矩阵（或者更直接一点，神经网络），是输入序列、是输出序列，是每一个时间步上通过前序时间步的输入和隐藏状态构建的新隐藏状态。这便是最原始的S4机制（Structured State Space for Sequences）的核心思想。
        有比较熟悉RNN的读者可能要问：这个式子本质上就是RNN的一种，它相比于RNN厉害在哪呢？我们回忆Transformer为什么能完爆RNN：训练速度。Transformer（或者说Attention模块）可以同时接收一个序列不同前缀长度的批量输入：
        并批量输出序列预测的结果
        因而具备相对高效的训练效率。而传统的RNN，每个隐藏状态都需要建立在前序时间步完成计算的情况下才能计算，而前序时间步又需要它的前序时间步完成计算，这样递归的依赖导致RNN在一个序列上只能串行计算。
        S4机制的天才之处在于，它把RNN式的计算转换成了CNN的形式。怎么理解呢？首先，对于任意从开始的时间步，有
        代入式
        可以递推得到：
        这组权重可以用一个1D卷积核来表示：
        这意味着：要计算输出序列，只需要使用输入序列，通过卷积核在时间轴上进行一次1D卷积就可以批量得到：
        所以，每次参数矩阵、和反向传播迭代后，只需要预制一个卷积核，就可以实现并行训练。
        细心的读者可能会发现：这个卷积核的计算，其实仍然涉及极高复杂度的计算。因为参数矩阵和均是维矩阵，在获取这个卷积核时，实际上仍然要进行复杂度的计算。
        有一类矩阵进行矩阵相乘的开销非常低，这就是对角矩阵（Diagonal Matrix）。假如我们直接把和都约束成对角阵，问题是不是就迎刃而解了？没那么简单，当强行约束为对角阵时，两个矩阵都丢失了大量的表示能力；并且，大量实验证明，Mamba的这两个参数矩阵必须使用特殊的HiPPO（High-order Polynomial Projection Operator）初始化，而这个HiPPO矩阵又是一个非常复杂的非对称高秩矩阵。
        Mamba的作者们非常天才地证明了，HiPPO矩阵具有DPLR（Diagonal Plus Low-Rank）性质，可以被完美分解成一个对角阵和一个低秩阵的和：
        其中，和均为列向量，二者乘积构成了一个秩为1的低秩矩阵。这个矩阵又能通过变换、Woodbury恒等式和iFFT的组合拳，快速求出卷积核。

      ### S6: Structured State Space Model with Selective Scan
        在S4的设计中，几个参数矩阵都是线性时不变的（Linear Time-Invariant，LTI）。S6的思想是，把参数矩阵全部定义为与输入有关的变量：
        但这带来的另一个问题是，由于参数矩阵的LTI性质被取消，上文讨论的那些快速运算的捷径被全部封杀了。

## MLLM大模型对比详情

| 模型 | 发布时间 | 关键改进 | 训练流程 |
|------|----------|----------|----------|
| GLM-4.1V/GLM-4.5V | 2025-07-01 | AIMv2-Huge 
3D-RoPE | 双阶段预训练 
SFT+RLCS后训练 |
| Qwen2.5-VL | 2025-01-26 | 动态分辨率 
动态帧率 
mRoPE多维编码 
绝对时间对齐 
Window Attention 
... | 三阶段预训练 
SFT+DPO后训练 |
| MiniCPM-V 4.5 | 2025-08-26 | 3D-Resampler | 三阶段预训练 
SFT+混合强化学习后训练 |
| Qwen3-VL | 2025-09-23 | MRoPE-Interleave 
DeepStack 
文本时间戳对齐 | GSPO |

### 详细对比

#### GLM-4.1V/GLM-4.5V

- **发布时间**: 2025-07-01
- **关键改进**: AIMv2-Huge 
3D-RoPE
- **训练流程**: 双阶段预训练 
SFT+RLCS后训练
- **数据工程**: 启发式与相关性筛选 
概念平衡重采样

#### Qwen2.5-VL

- **发布时间**: 2025-01-26
- **关键改进**: 动态分辨率 
动态帧率 
mRoPE多维编码 
绝对时间对齐 
Window Attention 
SwiGLU+RMSNorm
- **训练流程**: 三阶段预训练 
SFT+DPO后训练

#### MiniCPM-V 4.5

- **发布时间**: 2025-08-26
- **关键改进**: 3D-Resampler
- **训练流程**: 三阶段预训练 
SFT+混合强化学习后训练
- **数据工程**: OCR数据动态破坏

#### Qwen3-VL

- **发布时间**: 2025-09-23
- **关键改进**: MRoPE-Interleave 
DeepStack 
文本时间戳对齐
- **训练流程**: GSPO

