# 比赛经验

[[English]](https://github.com/ryanxingql/winner-ntire22-vqe/blog_en.md) [[项目主页]](https://github.com/ryanxingql/winner-ntire22-vqe)

作者：[幸群亮](https://ryanxingql.github.io/)

团队：郑美松，[樵明朗](https://github.com/MinglangQiao)，[蒋铼](https://github.com/remega)，[徐迈](https://scholar.google.com/citations?user=JdhDuXAAAAAJ)，刘怀达，[陈颖](https://scholar.google.com/citations?user=NpTmcKEAAAAJ)

- [比赛经验](#比赛经验)
  - [0. 赛事背景](#0-赛事背景)
  - [1. 最终方案](#1-最终方案)
  - [2. 准备数据集](#2-准备数据集)
    - [2.1. 自制线下验证集](#21-自制线下验证集)
    - [2.2. 类别均衡](#22-类别均衡)
  - [3. 选择基础模型](#3-选择基础模型)
  - [4. 改进基础模型](#4-改进基础模型)
    - [4.1. 失败的尝试](#41-失败的尝试)
      - [4.1.1. 模块化引入 transformer](#411-模块化引入-transformer)
  - [5. 优化训练方法](#5-优化训练方法)
    - [5.1. 模型前后端依次收敛](#51-模型前后端依次收敛)
    - [5.2. 三步收敛](#52-三步收敛)
    - [5.3. 删除重复帧](#53-删除重复帧)
    - [5.4. 通用 trick](#54-通用-trick)
  - [6. 优化测试方法](#6-优化测试方法)
    - [6.1. 删除重复帧](#61-删除重复帧)
    - [6.2. 通用 trick](#62-通用-trick)
  - [7. 模型级联](#7-模型级联)
  - [8. 后记](#8-后记)

## 0. 赛事背景

NTIRE 挑战赛全称 New Trends in Image Restoration and Enhancement workshop and challenges on image and video processing，是 CVPR 一系列 workshop 中较为著名的一个（[CVPRW 2022](https://cvpr2022.thecvf.com/workshop-schedule) 一共收录了约 71 个 workshop）。NTIRE 是 [Radu Timofte](https://scholar.google.ch/citations?user=u3MwH5kAAAAJ&hl=en) 主办的赛事，[第一次](https://data.vision.ee.ethz.ch/cvl/ntire/)在 ACCV 2016 举办，之后成为了 CVPRW 的常客。

<img src="figs/cvprw.png" width="60%">

图：CVPRW22

<img src="figs/ntire16.jpg" width="40%">

图：NTIRE16

NTIRE 挑战赛细分了多个项目。多帧质量增强大项是 2021 年开始由[杨韧](https://scholar.google.ch/citations?user=3NgkOp0AAAAJ&hl=en)和 Radu 组织的，[最初](https://data.vision.ee.ethz.ch/cvl/ntire21/)包含 “PSNR 质量优化”和“主观质量优化”两个主题。[今年](https://data.vision.ee.ethz.ch/cvl/ntire22/)主办方删除了主观质量优化赛道，只保留了 PSNR 质量优化赛道，同时引入了超分辨率、质量优化耦合赛道。

<img src="figs/ntire21.png" width="50%">

图：NTIRE21 多帧质量增强大项

<img src="figs/ntire22.png" width="50%">

图：NTIRE22 多帧质量增强大项

近年来中国互联网大厂不断涌入该赛事并宣传自己的比赛成果，赛事参与队伍越来越多，夺冠难度越来越大。21 年参赛队伍中包括北大、清华、复旦、南大、南洋理工 S-Lab、腾讯、Bilibili、字节跳动、大疆、华为诺亚、京东方等。今年又涌入了阿里巴巴、中兴、小米等企业，还有中科院自动化所、北航、哈工大、电子科大、南京理工、SIAT 等高校和研究院，以及港中文 XPixel、腾讯 GY-Lab 等知名实验室。

<img src="figs/participants.png" width="30%">

图：NTIRE21 参赛队伍

[北航 MC2 Lab](http://www.buaamc2.net/) 常年关注、深耕压缩视频质量优化技术。此次和阿里巴巴支撑淘宝视频业务的[淘系技术部](https://tech.taobao.org/)合作，强强联手，共同参赛。

## 1. 最终方案

- 数据
  - 赛事官方数据集，包含 240 个由 4K 降采样得到的 qHD 视频。
  - 从 YouTube 上搜集的 870 个 4K 高码率视频。

- 框架：两级模型级联

- 基础模型
  - 第一级：BasicVSR++（NTIRE21 冠军算法）
  - 第二级：SwinIR

- 模型改进
  - 第一级模型中的二阶传播改为 PQF 传播。
  - 第一级模型的后端 5 个 residual block 增加至 55 个。

- 训练方法优化
  - 模型前后端依次收敛：每次加 10 个 residual block，分 6 次训练和收敛。
  - 三步训练：首先在完整数据库上用 Charbonnier loss 收敛模型，然后改为 MSE loss 收敛一次，最后在官方数据集上收敛。
  - 删除 LQ 中的重复帧。

- 测试方法优化
  - x8TTA
  - 模型 ensemble
  - 删除 LQ 中的重复帧，增强后 copy 补全。

## 2. 准备数据集

多帧质量增强大项有一个特点：不约束选手使用的训练集。在 NTIRE21 赛事中，基本所有队伍都搜集了额外数据，其中夺冠的 B 站还使用了自家的 4K 视频。

<img src="figs/extra_data.png" width="80%">

图：NTIRE21 各队伍基本都使用了额外数据

今年为了防止选手根据测试集找数据，测试集视频只在赛事最后 7 天放出，且选手需要在放出测试集前上交模型和代码。

我们认真研究了 NTIRE21 的数据文档，严格按照去年的数据集设计模式从 YouTube 搜集数据。关键点：

- 视频类别：按照 NTIRE21 设置的 10 个分类搜集各 100 条 4-30s 视频，包括动物、自然、时尚、特写、运动、风景、人、城市和交通工具。
- 拍摄质量：我们发现主办方偏好 PGC（Professionally Generated Content）视频，例如纪录片、液晶演示片等。我们要求 PGC 占比超过 80%。
- 拍摄条件：要求 20% 暗光，20% 快速运动（如直升机），30% 手持拍摄视频。
- 拍摄视角：要求 40% 平视，20% FOV（field of view，如鱼眼、大光圈）、20% 航拍、20% 仰视。
- 帧率：低帧率、高帧率视频各占一定比例。
- 高画质：要求高码率，尽量减少拍摄噪声和传输压缩造成的干扰；要求画质清晰（明亮通透，纹理清晰）。
- 低冗余：可以从一个视频里切至多 4 个片段。
- 无转场。

和外包不断沟通，肉眼核查每条视频，返回意见。要求搜集 1000 条，实际搜集 1600 条（含不合格）。这为 2.2 节的类别均衡埋下了伏笔。

额外数据的增益在 0.1-0.2 dB。

### 2.1. 自制线下验证集

赛事验证集没有提供 ground truth；我们需要在网站上提交我们增强的结果，得到一个分数。这样有两个问题：

1. 我们无法频繁提交：时间成本大，提交次数有限。
2. 我们无法知道模型在每一个视频、每一帧上的结果，而只能知道一个平均分。

因此，我们严格按照 10 个分类，选择了 10 个视频作为验证集。其中大部分都是去年测试集视频。

<img src="figs/validation_offline.png" width="80%">

图：使用自制验证集验证算法

自制验证集在比赛后期非常好用：

1. 方便快捷：官方验证集有 15 个视频，我们只有 10 个视频，测试时间更短；不需要提交，可以自己算 PSNR。
2. 准确：因为选择得当，因此我们的结果和官方验证集结果的趋势是完全一致的。我们可以放心大胆地根据自制验证集来迭代模型。
3. 详细：能知道模型在不同分类上的表现。

<img src="figs/psnr_categories.png" width="50%">

图：使用自制验证集查看不同分类表现

### 2.2. 类别均衡

前面提到，我们动用外包资源，搜集了 1600 条数据。原本我们计划一个分类搜集 100 条视频，但因为外包数据存在不合格数据（相似度太高，有噪声等），因此实际获取的视频数量更大。

一开始，我们将尽可能多的视频扔进了训练集。在比赛后期我发现，每个分类视频数量非常不均匀：例如城市分类只有 87 个视频，而时尚分类有 200+ 视频。而时尚分类大多是走 T 台的室内、暗光、黑色背景视频，质量较差。因此，我们以视频数量最少的城市分类为基准，每个类别随机选取了 87 个视频，一共选取了 870 个视频参与训练。经过这样非常简单的均衡化处理，模型表现提升 0.02 dB。

Btw 大家不要小瞧 0.02 dB。在三个赛道的最终测试集上，我们分别比腾讯 GY-Lab 高 0.009 dB、高 0.101 dB、低 0.012 dB。

<img src="figs/final_results.png" width="50%">

图：最终结果

## 3. 选择基础模型

打比赛有两条路：

1. 提出颠覆性的网络结构，例如 ResNet、MPRNet、EDVR、BasicVSR。
2. 基于基础模型进行升级和调参。

一个好的网络结构往往能主导数年的竞赛方案，例如去年至少有两支队伍使用了 BasicVSR，两支队伍使用了 EDVR，都取得了不错的成绩。我们选择了第二种方案。

无论选择哪一种方案，我们都会选择一个基础模型，在此基础上进行改进，来获得我们的比赛模型。另起炉灶非常浪费时间。就好比，我们要参加 NTIRE23 的比赛，我们一定会把 NTIRE22 的算法和预训练模型拿来接着用，而不是从基础代码开始写。

在选择基础模型时，我们遇到了一个问题：要不要随大流，选择一个基于 transformer 的网络结构。我们比较担心今年的对手会采取类似的策略，在算法学习能力上和我们拉开差距。

经过大量调研，我们分析了 transformer 的可行性和可靠性：

- 显存不足：Video transformer 中的代表 VRT 使用了 8 卡 80 GB A100。而我们的计算资源为 1 台 8 卡 32GB V100 服务器和若干台 4 卡 32GB V100 服务器。
- 耗时太长：VRT 一个模型要训练 5-7 天左右；而 A100 的速度是 V100 的 3 倍，因此我们训练一个相同的模型需要至少半个月。
- 资源有限下性能堪忧：根据 VRT 的报告，在 7 帧短视频（如 Vimeo 数据集）上，VRT 性能超过 BasicVSR++；但对于长视频（如 REDS 每个视频 100帧），由于显存受限，VRT 无法一次性输入超过 16 帧的视频，性能上无法超过 BasicVSR++（能一次性输入 30 帧以上）。

<img src="figs/vrt.png" width="80%">

图：VRT 实验报告

因此，我们决定基础模型仍然采用基于 RNN（实际上是 CNN）的 BasicVSR++。当然选择 BasicVSR++ 还有以下原因：

- 性能出色（在 NTIRE21 中仅借助 REDS 获得第二名）。
- 复杂度较高（单卡单个最小 256x256 patch 显存约 22 GB）；其他小模型算法没有经过比赛验证，我们不敢拿来用。
- 开源质量好（训练代码未开源，但测试代码集成到了 mmediting），代码逻辑清晰。
- 预训练模型可以直接拿来 fine-tune，非常省时间。要知道从头训练一个 BasicVSR++ 模型也需要一周时间。

尽管如此，我们也没有完全放弃 transformer，而是决定在后续模块化尝试 transformer 结构。参见 4.1.1 节。

关于第二级基础模型，参见第 7 节。

## 4. 改进基础模型

为了更好地改进基础模型，我们必须要对基础模型的提出者进行深入的研究。因为在他们常年的探索过程中，一定会有心得体会和关键成果。

以 BasicVSR 提出者 Kelvin 为例。

- Kelvin 在 2021 年 AAAI 一篇[文章](https://arxiv.org/abs/2009.07265)中研究了 EDVR 的性能瓶颈。在这篇文章中，Kelvin 告诉大家，虽然在特征域上进行时序对齐的 DCN 要比传统的、在像素域上进行对齐的光流方法更强大，但他们的学习目标和效果几乎是一样的。那么，既然端到端学习时 DCN offset 如此不稳定、难学，我们就单独给 DCN offset 加一个损失函数，要求它和光流输出保持一致。通过额外的监督，就稳定了 DCN offset 的学习。

<img src="figs/unstable_dcn.png" width="35%">

图：EDVR 中 DCN 溢出通常发生在 300K iterations，此时模型 loss 也不降反升

<img src="figs/similarity_flow_dcn_offset.png" width="50%">

图：Flow 和 DCN offset 的形态几乎一模一样，这为后面 DCN offset 的稳定性改进埋下了伏笔

- Kelvin 在 BasicVSR++ 中进一步改善了这个策略，即将光流输出作为 base，额外学习一个 residual，然后把 base 和 residual 加起来作为 DCN 的 offset。

<img src="figs/flow_based_dcn_offset.png" width="50%">

图：BasicVSR++ 中的 flow-based DCN offset

从以上研究历程中我们得到两点关键：

- 对齐在视频增强任务中起到了非常重要的作用。
- 我们不能再沿用 EDVR 或 STDF 的 DCNv2，而是使用更为先进的、基于 flow 的 DCN 方法。

再以我们实验室工作为例。

- 在 19 年 ICME 一篇[文章](https://arxiv.org/abs/1903.04596)中，杨韧研究了 LSTM 中不同帧的参与度，结果 PQF 确实起到了非常重要的作用。

<img src="figs/pqf_contribution.png" width="50%">

图：PQF 在 LSTM 质量增强任务上起到了非常重要的作用

- 在本人 19 年 [MFQEv2](https://arxiv.org/abs/1902.09707) 溶解试验中，我尝试过将相邻帧输入滑窗网络，结果增强性能剧烈下降，远不如输入 PQF 的网络。

<img src="figs/without_pqf.png" width="50%">

图：PQF 在滑窗网络质量增强任务上同样非常重要

因此，PQF 一定会在压缩视频质量增强任务中起到关键作用，必须用。

再回到我们的基础模型。虽然 BasicVSR++ 采用了双向传播结构，理论上每一帧增强时都能获得来自双向 PQF 的信息；但我认为这样做还不够，我们应该让模型能直接从 PQF 中获取有用的信息。因此，我提出用 PQF 替换 BasicVSR++ 中的二阶相邻帧。模型表现提升 0.05 dB。

<img src="figs/second_order_propagation_basicvsrpp.png" width="60%">

图：BasicVSR++ 中的两阶传播

<img src="figs/pqf_propagation.png" width="60%">

图：PQF 传播

此外，我们将重建部分原本 5 个 residual block 增加至 55 个。关于训练方法，参见 5.1 节。

### 4.1. 失败的尝试

除此之外我们还尝试过大量的网络结构优化，大多都失败了。例如，我们仿照 [IconVSR](https://arxiv.org/abs/2012.02181)，在 PQF 上额外建立了 refill 支路；此外，我们仿照 [RealBasicVSR](https://arxiv.org/abs/2111.12704)，对输入图像进行 pre-cleaning；我们把这些设计应用到了 BasicVSR++ 这种大模型上，没有产生任何效果。

<img src="figs/refill_key_frame.png" width="50%">

图：对关键帧的特征加强提取（refill）

<img src="figs/precleaning.png" width="50%">

图：对输入图像预去噪再提取特征

我个人认为，有很多论文中有效的方法，很有可能只是因为计算量上去了。当我们在复杂度足够大的网络上实践时，往往就丢掉了魔力。这也是为什么我在选择基础模型的时候强调，一定要选择复杂度足够高且证明有效的网络结构。

此外，我们注意到 BasicVSR++ 网络结构呈现梨形结构：两头简单，中间复杂。我们尝试过减少时序部分的块数，增加两头、尤其是重建部分的块数，但效果都变差了。

<img src="figs/shallower_propagation.png" width="35%">

图：改善梨形分布

但实验也没白做，至少有两点启发：

1. 特征提取不能太复杂，否则特征太干净，丢掉了细节。这一点和 Kelvin 最新的 [RealBasicVSR](https://arxiv.org/abs/2111.12704) 结论一致。
2. 时序传播非常重要，少一点块都会导致较大的性能损失。

另外，我们还尝试了 ConvNeXt 等在高层视觉领域被广泛认可的网络新结构，但在底层视觉领域效果不佳。此外，我们发现，RDN 等复杂结构的效果往往不如 residual block。因此我们保留了原始 residual block 堆叠设计。

<img src="figs/convnext.png" width="80%">

图：在高层视觉任务上超越 Swin 和 ResNet 的 ConvNeXt

#### 4.1.1. 模块化引入 transformer

仿照 SwinIR，我将相当比例的 residual block 替换为 Swin block。尝试了各种办法，例如从后往前换，从前往后换，等等等等。但都没有突出效果，反而显存紧张了许多，计算时间也因为自注意力计算变长了。

<img src="figs/transformer_block.png" width="80%">

图：尝试过的各种方案

在这些方案中，用 Swin block 替换前端模块效果最差，替换后端模块效果最好。我们怀疑替换前端模块导致无法加载预训练参数，在训练初期有非常大的误差传播；相反，替换后端模块的影响就小得多。

我们猜测，transformer 要用得好，必须得在大规模数据库上预训练过（根据其他论文）。直接 copy 一个 Swin block 过来从头训练，很难达到 CNN 的效果。关于 transformer 的成功应用，我们在讲第二级模型时再说。

我尝试将其中最好的版本（代号为 swinv2p4-5k）和没有 transformer 改进的模型（代号为 v4p6-35k）进行模型 ensemble，即把二者的结果取平均。结果比二者 PSNR 都高。这也算一个成功的尝试。但前提是二者 PSNR 不能差距太大。

<img src="figs/ensemble.png" width="30%">

图：两个模型 ensemble 产生了比两个模型都好的结果

## 5. 优化训练方法

### 5.1. 模型前后端依次收敛

直接训练一个大模型是非常困难的。我们最终模型有多达 7000 万参数，规模非常庞大。且网络中每一个模块前后依赖，如果训练初期前端模块没有收敛，那么后端模块的输入就是有问题的，自然无法收敛。因此，直接端到端训练是很难得到好模型的。事实上，模型前后端依次收敛也是 Kaggle 比赛中的常用、甚至必用的方法，其目的就是让大模型的每一个模块前后依次收敛。

我们每一次增加 10 个 residual block，从 5 加到 55（此时 32GB V100 显存打满），一共 6 次训练和收敛。这样一套下来，训练成本非常高，但增益也很明显，和直接端到端训练大模型相比，提升约 0.1 dB。

<img src="figs/progressive_training.png" width="80%">

图：在重建模块每次增加 10 个 residual block，不固定参数；每次都要加载上一次训练好的参数

### 5.2. 三步收敛

我们都知道，MSE 和 PSNR 只差一个对数关系。因此，优化 MSE 本质上等价于优化 PSNR。但奇怪的是，NTIRE21 中近半队伍都使用 Charbonnier 损失函数。

我们首先做了实验，发现 Charbonnier 损失函数下模型收敛速度比 MSE 模型快很多，甚至 PSNR 也更高。大量文章佐证了这个观点，我在这里截取一篇[文章](https://www.mdpi.com/2079-9292/10/11/1234/pdf)中的图：

<img src="figs/different_losses.png" width="50%">

图：Charbonnier loss 在 SISR 任务上的 PSNR 表现比 L2 还好

但我们不信邪。我们先使用 Charbonnier loss，等模型收敛后，再使用 L2 loss，模型表现进一步提升 0.02 dB。

最后，我们考虑到自制数据集可能和官方数据集的制作流程有一定偏差，且官方数据集更能反映主办方对数据的偏好，因此我们将收敛的模型在官方数据集上用 L2 loss 做最后的 fine-tune，模型表现进一步提升 0.02 dB。

### 5.3. 删除重复帧

我们在制作数据集时意外发现，无论是官方数据集还是我们的额外数据集，大约有 30% 的视频存在重复帧。

这些视频的 GT 中并没有重复帧，但因为相邻两帧变化较小（相邻 PSNR 在 50-70 dB），在压缩以后就变得完全相同。

<img src="figs/repeated_frame.png" width="60%">

图：重复帧现象

前面我们提到，时序信息在质量增强任务中发挥了至关重要的作用。BasicVSR 表明，输入帧数下降，性能也随之下降；而 VRT 之所以没有超过 BasicVSR++，一个很重要的原因正是显存有限、只能处理 16 帧视频，而 Basic 能处理 30 帧。

<img src="figs/frame_number.png" width="50%">

图：BasicVSR 实验中，一个视频中切片越多，每个片段帧数越小，PSNR 就越差

在我们看来，LQ 视频中这些重复帧对网络而言就是冗余。如果我们删除重复帧，那么有效帧数就提高了，网络性能也能随之提高？

我们先只在测试阶段删除了重复帧，在增强后通过直接 copy 的方式恢复重复帧，效果几乎没变，稍微差一点点。

我们随后尝试在训练时删除重复帧，结果模型学习效果越来越出色，增益 0.05-0.1 dB 左右。

<img src="figs/repeated_frame_enh.png" width="70%">

图：处理重复帧方案

4/17/22：我们测试了其他数据集，发现模型表现不稳定，有时会差 0.2 dB。这个方法和数据关系很大，谨慎使用。

### 5.4. 通用 trick

加载预训练模型。Kaggle 大神 Pavel Ostyakov 的经历：在一个 NLP 比赛中，因为数据结构比较像图像，就加载了 ImageNet 预训练模型，结果效果非常好。

Warmup。有两点好处：

1. 在训练初期，loss 很大，较小的学习率可以缓解过大的梯度，防止大模型溢出。BasicVSR++ 模型就遇到了这个情况，当学习率为 1e-4 时，训练没多久就会 NaN。
2. 学习率是主观设置的，你不知道学习率应该设置多大。通过观察 warmup 期间 loss 的下降幅度，你能大概判断出学习率多大比较合适，有利于 loss 快速下降。

<img src="figs/lr.png" width="80%">

图：典型的学习率变化曲线

Scheduler。就算是我们用了 Adam、AdamW 等一些带自适应和衰减的优化器，我们也要用学习率 scheduler。训练末期学习率如果保持在较高水平，模型很难收敛。

还有其他一些 trick 我们没有尝试，原则都是让模型从简到难学习，像课程学习一样：

- 逐步 augmentation：训练初期不要开随机翻转、旋转（data augmentation），随着训练进行，逐步打开。
- 从小分辨率开始：对分类任务很有帮助。

注意上述步骤最好不要一起做，因为会有叠加效应。应该分阶段分别做。

## 6. 优化测试方法

### 6.1. 删除重复帧

如果模型训练时没有删除重复帧，那么测试时删除重复帧几乎没有增益，反而稍微变差了一点。

如果模型训练时也删除了重复帧，那么测试时删除重复帧会带来 0.05 dB 左右的增益。测试视频中大概有 3-5 个视频存在重复帧。

### 6.2. 通用 trick

模型 ensemble：前面提到了。最简单、最安全的办法是两个模型的结果直接取平均，要求两个模型的性能接近。

x8tta：翻转、旋转输入图像，分别输入模型，将得到的结果恢复原始形态，再取平均。因为是测试阶段执行的 data augmentation，不需要训练，因此称为 test time augmentation（tta）；4 种旋转（0、90、180、270 度）和 3 种翻转（不翻转，y 对称翻，x 对称翻）理论上有 12 种组合，但去除重复组合后只剩 8 种，因此称为 x8tta。模型表现直接提升 0.1-0.2 dB。

<img src="figs/x8tta.png" width="50%">

图：x8tta，引自 Radu 的[一篇文章](https://arxiv.org/abs/1511.02228)

## 7. 模型级联

<img src="figs/diagram.png" width="60%">

图：两级模型级联

级联是一定要做的。有大量的文章指出，与其训练一个超级大模型，不如训练多个小模型，要么级联，要么做 ensemble。

我们尝试过级联两个 BasicVSR++ 模型，结果发现第二级模型会迅速收敛，但表现类似一个 auto-encoder（AE）：即输出基本等于输入，PSNR 表现和第一级模型差异小于 0.001 dB。我们怀疑第二级模型不宜采用一个和第一级相同结构的模型。具体原因没有深究。

在后续的实验中，我们尝试了 [MPRNet](https://arxiv.org/abs/2102.02808) 和 SwinIR，效果都很不错，起码不再是一个 AE 了。

选择 MPRNet 有两个理由：

1. MPRNet 是众多竞赛模型的基础模型，广泛用于去模糊等底层视觉任务。
2. MPRNet 有多输入；我们希望在第二级模型引入原始输入，重新引入第一级 L1 训练丢掉的纹理细节。

<img src="figs/mprnet.png" width="50%">

图：MPRNet 的多输入结构

后面我们尝试了 SwinIR，效果更好，因此转向了 SwinIR。我们使用了 SwinIR 的 color denoise 预训练模型，从而能在半天内训练一个模型。这再次证明了 transformer 结构预训练的重要性。

## 8. 后记

我们做到了以下很重要的几点：

- 规划和合作。打比赛是一个系统工程，每一步都需要计划、分工、协作、探讨。越到后期越觉得，我们团队少一个人都不行，每个人的付出都特别关键。
- 取胜的信念。拿一个好的名次并不难，要有信心，特别是在比赛末期大伙刷榜的时候。
- 充分的实验。全组在探索阶段做了超过 200 组实验，有很多改进点是从实验中摸索出来的。拍脑门想出来的方案大多不 work。
