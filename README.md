# How We Win the NTIRE 2022 Challenge on Super-Resolution and Quality Enhancement of Compressed Video

Our team (TaoMC2) wins the [NTIRE 2022 challenge on super-resolution and quality enhancement of compressed video](https://data.vision.ee.ethz.ch/cvl/ntire22/)!

- Track 1 (Quality enhancement): **rank 1st!**
- Track 2 (Quality enhancement and x2 SR): **rank 1st!**
- Track 3 (Quality enhancement and x4 SR): **rank 2nd!**

## Open Source

- [x] Experience sharing: [[Blog-zh]](https://github.com/ryanxingql/winner-ntire22-vqe/blob/main/blog_zh.md) [[Blog-en]](https://github.com/ryanxingql/winner-ntire22-vqe/blob/main/blog_en.md)
- [x] Performance report: [[NTIRE22 report]](https://arxiv.org/abs/2204.09314)
- [x] Method report: [[CVPR workshop paper]](https://arxiv.org/abs/2204.09924)
- [x] Inference model: [[README]](https://github.com/ryanxingql/winner-ntire22-vqe/blob/main/README_test.md)

NOTE: Our approach consists of two stages and x8 test time augmentation (TTA) operations. One can infer only the first stage of our model without TTA and achieve good performance. One can also run the full inference of our approach to achieve optimal performance, but with more memory and longer running time.

## Acknowledgments

Thanks to the great efforts of the open-sourced projects [MMEditing](https://github.com/open-mmlab/mmediting) and [SwinIR](https://github.com/JingyunLiang/SwinIR).

If you find this work helpful, please cite our workshop paper.
