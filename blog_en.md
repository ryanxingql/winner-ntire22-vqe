# Experience of Winning the NTIRE 2022 Video Enhancement Challenge

[[中文]](https://github.com/ryanxingql/winner-ntire22-vqe/blob/main/blog_zh.md) [[Project Page]](https://github.com/ryanxingql/winner-ntire22-vqe)

- [Experience of Winning the NTIRE 2022 Video Enhancement Challenge](#experience-of-winning-the-ntire-2022-video-enhancement-challenge)
  - [1. Challenge Background](#1-challenge-background)
  - [2. Challenge Results](#2-challenge-results)
  - [3. Our Solution](#3-our-solution)
    - [3.1 PQF Propagation](#31-pqf-propagation)
    - [3.2 Remove Duplicated Frames](#32-remove-duplicated-frames)
    - [3.3 Three-Step Convergence](#33-three-step-convergence)
    - [3.4 Progressive Training](#34-progressive-training)
  - [4. Thinking](#4-thinking)
    - [4.1 Why Not Use the Video Transformer](#41-why-not-use-the-video-transformer)
    - [4.2 In-Depth Research on Long-Term Works](#42-in-depth-research-on-long-term-works)
    - [4.3 Select a Baseline with High Complexity](#43-select-a-baseline-with-high-complexity)
    - [4.4 Use Transformers with Pre-training](#44-use-transformers-with-pre-training)
    - [4.5 Postscript](#45-postscript)

Authors: [Qunliang Xing](https://ryanxingql.github.io/), [Minglang Qiao](https://github.com/MinglangQiao), [Lai Jiang](https://github.com/remega) and [Mai Xu](https://scholar.google.com/citations?user=JdhDuXAAAAAJ)

Affiliation: [Beihang MC2 Lab](http://www.buaamc2.net/)

## 1. Challenge Background

The full name of the NTIRE Challenge is "New Trends in Image Restoration and Enhancement workshop and challenges on image and video processing". The NTIRE challenge is one of the most famous CVPR workshops (A total of about 71 workshops are included in [CVPRW 2022](https://cvpr2022.thecvf.com/workshop-schedule)). NTIRE is a workshop hosted by [Radu Timofte](https://scholar.google.ch/citations?user=u3MwH5kAAAAJ&hl=en), which is first held at [ACCV 2016](https://data.vision.ee.ethz.ch/cvl/ntire/) and has since become a regular at CVPRW.

<img src="figs/cvprw.png" width="60%">

Figure: CVPRW22.

The NTIRE challenge is broken down into multiple challenges. The multi-frame quality enhancement challenge is organized by [Yang](https://scholar.google.ch/citations?user=3NgkOp0AAAAJ&hl=en) and Radu starting in 2021, which [initially](https://data.vision.ee.ethz.ch/cvl/ntire21/) contains two main tracks: fidelity (PSNR) optimization and perceptual quality (MOS) optimization. The organizers remove the perceptual track [this year](https://data.vision.ee.ethz.ch/cvl/ntire22/) and introduce the super-resolution and enhancement tracks.

<img src="figs/ntire21.png" width="60%">

Figure: NTIRE21 multi-frame quality enhancement challenges.

<img src="figs/ntire22.png" width="60%">

Figure: NTIRE22 multi-frame quality enhancement challenges.

In recent years, more and more teams have been pouring into these challenges and publicizing their titles. It has become more and more difficult to win the championship. The participating teams in NTIRE21 include Peking University, Tsinghua University, Fudan University, Nanjing University, NTU S-Lab, Tencent, Bilibili, ByteDance, DJI, Huawei Noah, BOE, etc.

<img src="figs/participants.png" width="30%">

Figure: NTIRE21 teams.

This year, more teams from Alibaba, ZTE, Xiaomi, XPixel, Tencent GY-Lab, the Institute of Automation of the Chinese Academy of Sciences, Beihang University, Harbin Institute of Technology, University of Electronic Science and Technology, Nanjing University of Science and Technology and SIAT have also participated.

[Beihang MC2 Lab](http://www.buaamc2.net/) has been focusing on video quality enhancement methods for many years. This year, we participate in this challenge for the first time.

## 2. Challenge Results

We won two championships and one runner-up in all three tracks.

<img src="figs/results.png" width="80%">

Figure: Challenge results.

Our approach showed strong generalization ability in the Track 2. We won the first place with an advantage of 0.1 dB.

## 3. Our Solution

Based on our baseline model BasicVSR++, we discovered and implemented four key innovations to achieve our results.

### 3.1 PQF Propagation

PQFs (key frames) refer to the frames with high PSNR score in the video, which are usually given high bit rates or small quantization steps. The distribution of PQFs is as follows: the first frame is PQF, followed by another PQF every 3 non-PQFs. If we denote PQFs by 1 and non-PQFs by 0, then the video frames can be represented as: 1000 1000 1000...

BasicVSR++ takes both forward and backward propagations. In theory, each frame can receive information from two PQFs in both sides. However, this information sometimes takes multiple steps of propagation to arrive. Taking the first non-PQF as an example, its right-hand second-order propagation is still from a non-PQF. Therefore, we propose to replace the second-order propagation in BasicVSR++ with PQF propagation.

<img src="figs/second_order_propagation_basicvsrpp.png" width="60%">

Figure: Propagation in BasicVSR++

<img src="figs/pqf_propagation.png" width="60%">

Figure: PQF propagation

Experiments show that the model performance is improved by 0.05-0.1 dB.

### 3.2 Remove Duplicated Frames

We discovered by accident that about 30% of the videos had duplicated frames, both for the official data-set and our data-set.

There are no duplicate frames in the ground truth videos; but because of the high similarity of adjacent two frames (50-70 dB between two frames), they become identical after compression.

<img src="figs/duplicated_frame.png" width="60%">

Figure: Duplicated frames.

The temporal information and alignment play a key role in quality enhancement (to be mentioned in Section 4.2). BasicVSR shows that as the number of input frames decreases, the performance also decreases; and a very important reason why VRT does not exceed BasicVSR++ is that it can only handle 16 frames of video, while BasicVSR++ can handle 30 frames.

<img src="figs/frame_number.png" width="50%">

Figure: In the BasicVSR experiment, the more slices in a video, the smaller the number of frames per slice, and the worse the PSNR.

In our opinion, these duplicated frames in LQ videos are redundant to the network. If we remove duplicated frames, then the effective number of frames increases and the network performance may improve.

<img src="figs/duplicated_frame_enh.png" width="70%">

Figure: How we deal with duplicated frames.

We first tried to remove the duplicated frames in the testing phase. We remove the duplicated frames before enhancement and restore the duplicated frames by copying after enhancement. The result slightly decreased.

We then tried to remove duplicated frames during both the training and testing, and obtained a gain of around 0.05-0.1 dB.

### 3.3 Three-Step Convergence

We all know that optimizing MSE is equivalent to optimizing PSNR. But strangely, nearly half of the teams in NTIRE21 use the Charbonnier loss function.

We first did experiments and found that the model with Charbonnier loss converged much faster than the model with MSE loss. Moreover, the PSNR can be higher for the model with Charbonnier loss. A large number of articles corroborate this finding, like [this](https://www.mdpi.com/2079-9292/10/11/1234/pdf):

<img src="figs/different_losses.png" width="50%">

Figure: PSNR performance with Charbonnier loss on SISR task is better than that with L2 loss.

We used a three-step training method with success:

1. Training over our data-set with Charbonnier loss.
2. Training over our data-set with L2 loss. The model performance is improved by 0.02 dB.
3. Training over the official data-set with L2 loss. The model performance is further improved by 0.02 dB.

We guess that the official data-set can better reflect the organizers' preference for data; besides, our data generation process may have a certain deviation from the official one. Therefore, we conduct the third step of training.

### 3.4 Progressive Training

It is very difficult to directly train a large model. Our final model has up to 70 million parameters. Each module in the network relies on the previous module. If the front-end module does not converge in the early stage of training, then the input of the back-end module is problematic. Therefore, direct end-to-end training is hard to obtain a good model. Therefore, we need to progressively train and converge each module of our large model.

We added 10 residual blocks for reconstruction each training time, i.e., from 5 to 55 blocks (until the 32GB V100 memory is filled up). Therefore, we took a total of 6 times of training.

<img src="figs/progressive_training.png" width="80%">

Figure: 10 residual blocks are added to the reconstruction module each time, and the parameters are not fixed; the previously trained parameters should be loaded.

In this way, the training cost becomes very high, but the gain is also significant. Compared with the direct end-to-end training of our large model, the improvement is about 0.1 dB.

## 4. Thinking

### 4.1 Why Not Use the Video Transformer

When choosing the baseline model, we thought about whether to choose a transformer-based network structure. We analyzed the feasibility and reliability of the transformer, especially the representative VRT method, and found three main issues:

- We have insufficient GPU memory: VRT uses 8 A100 GPUs with 80 GB memory each. Instead, we have only one machine with 8 V100 GPUs of 32 GB memory and several machines with 4 V100 GPUs.
- Training transformers is time-consuming: VRT takes about 5-7 days to train a model, and the speed of A100 is 3 times that of V100, so it takes at least half a month for us to train the same model.
- The performance of the video transformer is limited: According to VRT's report, on 7-frame short videos (such as Vimeo data-set), VRT's performance exceeds that of BasicVSR++; but for long videos (such as REDS with 100 frames per video), VRT cannot input more than 16 frames at one time due to the limited video memory, while BasicVSR++ can input more than 30 frames; as a result, the performance of VRT is lower than that of BasicVSR++.

<img src="figs/vrt.png" width="80%">

Figure: VRT experiment report.

Therefore, we did not use the video transformer as our baseline model.

### 4.2 In-Depth Research on Long-Term Works

To improve our baseline model, we conducted in-depth research on some long-term works in this field, since there must be experiences and key findings in these works.

Let's take the works of Kelvin (the author of BasicVSR) as examples.

Kelvin investigates the bottleneck of the EDVR model in his 2021 AAAI [article](https://arxiv.org/abs/2009.07265). In this article, Kelvin shows us that although DCNs with temporal alignment in the feature domain are more powerful than the traditional optical flow methods in the pixel domain, the learned flows and DCN offset are almost the same. Then, since the DCN offset is so unstable and difficult to train during end-to-end learning, we can add a loss function to the DCN offset, requiring it to be consistent with the optical flow. With additional supervision, the learning of the DCN offset is stabilized.

<img src="figs/unstable_dcn.png" width="35%">

Figure: The overflow of DCN in EDVR usually occurs at 300K iterations, at which time the model loss rises.

<img src="figs/similarity_flow_dcn_offset.png" width="50%">

Figure: The learned flow and DCN offset are almost the same, which lays the groundwork for the stability improvement of DCN offset in BasicVSR++.

Kelvin further improved this strategy in BasicVSR++, that is, the optical flow is used as the base, an additional residual is learned, and then the base and the residual are added to become the offset of the DCN.

<img src="figs/flow_based_dcn_offset.png" width="50%">

Figure: flow-based DCN offset in BasicVSR++.

From the above, we have two key points:

- Alignment plays a very important role in video enhancement tasks.
- We should not use the DCN in EDVR and STDF, but use the more advanced flow-based DCN in BasicVSR++.

Next, take our lab works as examples.

In Yang's [article](https://arxiv.org/abs/1903.04596), Yang studied the contributions of different frames in LSTM and showed that PQF did play a very important role in video enhancement.

<img src="figs/pqf_contribution.png" width="50%">

Figure: PQF plays a very important role in the LSTM-based quality enhancement.

In my [MFQEv2](https://arxiv.org/abs/1902.09707) ablation, I tried feeding adjacent frames into a sliding window network, and the result dramatically dropped, far worse than feeding PQFs.

<img src="figs/without_pqf.png" width="50%">

Figure: PQF is also very important for sliding window networks.

Therefore, PQFs play a key role in the task of compressed video quality enhancement.

### 4.3 Select a Baseline with High Complexity

We have tried many other improvements, but most of them have failed. For examples, we followed [IconVSR](https://arxiv.org/abs/2012.02181) to build an additional refill branch on PQF; we followed [RealBasicVSR](https://arxiv.org/abs/2111.12704) to pre-clean the input image. These methods were applied to our large model and had no effect.

<img src="figs/refill_key_frame.png" width="50%">

Figure: Refill key frames.

<img src="figs/precleaning.png" width="50%">

Figure: Pre-clean the input image.

In my view, some methods are effective in paper mainly due to the increment of complexity. When we practice these methods on a complex big model, the magic of these methods disappears. That is why we should choose a baseline model that is complex enough and is proven to be effective in competitions.

### 4.4 Use Transformers with Pre-training

Following SwinIR, we replaced a few residual blocks with Swin blocks. We have tried various replacing methods but get no gain.

Among these methods, replacing front-end blocks with Swin blocks did the worst, and replacing back-end blocks worked best. We suspect that replacing front-end blocks results in large error propagation in the early stages of training; conversely, replacing back-end blocks has less impact.

We also believe that to use the transformers well, the transformers must be pre-trained on a large data-set. It is difficult to achieve the effect of CNNs by directly copying a Swin block and training them from scratch.

### 4.5 Postscript

In my view, our winning is mainly due to the following points:

- Planning and cooperation. Winning a competition is a big project, of which every step requires planning, division of labor, collaboration, and discussion.
- The belief to win. It's not difficult to get a good ranking; you have to be confident.
- Large numbers of experiments. We have done hundreds of sets of experiments, from which many improvements were inspired and validated. On the contrary, most of our ideas without experimental findings didn't work.
