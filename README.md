# Introduction
Code is based on Fairseq 0.6.2, and readme is adapted from the origin readme.
- **parallel multi-scale attention(MUSE)**
  - Code for [Zhao et al. (2019): MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning ](https://arxiv.org/abs/1911.09483)
  - [Pre-trained models as well as instructions](examples/parallel_multi-scale_attention(MUSE)/README.md) on how to train MUSE

# TL;DR:
A simple module consistently outperforms self-attention and Transformer model on main NMT datasets with SoTA performance.

# Three questions in sequence to sequence learning:

 - Is attention alone good enough？

 - Is parallel representation learning applicable to sequence data and tasks?

 - How to design a module that combines both inductive bias of convolution and self-attention？

# Key features:

- First successfully combine convolution and self-attention in one module for sequence tasks by the proposed shared projection,

- SOTA on three main translation datasets, including WMT14 En-Fr, WMT14 En-De and IWSLT14 De-En,

- Parallel learn sequence representations and thus have potential for acceleration.


# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

**Installing from source**

To install MUSE from source and develop locally:
```
pip install --editable . --user
```

# Pre-trained models and examples

We provide pre-trained models and detailed example training and
evaluation in [examples/parallel_multi-scale_attention(MUSE)/README.md](examples/parallel_multi-scale_attention(MUSE)/README.md).

# Results

| Task | size  | test (BLEU) |
| ---------- | ---:| ----:|
| IWSLT14 De-En | Base | 36.3 |
| WMT14 En-De |  Large  | 29.9 |
| WMT14 En-Fr |  Large | 43.5 |

# License
MIT-licensed.
The license applies to the pre-trained models as well.
We also provide an additional patent grant.

# Citation

Please cite as:

```bibtex
@article{zhao2019muse,
  title={MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning},
  author={Zhao, Guangxiang and Sun, Xu and Xu, Jingjing and Zhang, Zhiyuan and Luo, Liangchen},
  journal={arXiv preprint arXiv:1911.09483},
  year={2019}
}
```
