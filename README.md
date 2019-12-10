# Introduction


**Relevent links:**
 - [Arxiv pdf](https://arxiv.org/abs/1911.09483): https://arxiv.org/abs/1911.09483
 - [Pre-trained models as well as instructions for training](examples/parallel_multi-scale_attention(MUSE)/README.md): examples/parallel_multi-scale_attention(MUSE)/README.md
 - [Reddit post link](https://www.reddit.com/r/MachineLearning/comments/e13qhb/r_a_simple_module_consistently_outperforms/)

**About the paper:**

TL;DR: A simple module consistently outperforms self-attention and Transformer model on main NMT datasets with SoTA performance.

We ask three questions:
 - Is attention alone good enough？
 - Is parallel representation learning applicable to sequence data and tasks?
 - How to design a module that combines both inductive bias of convolution and self-attention？

We find that there are shortcomings in stand-alone self-attention, and present a new module that maps the input to the hidden space and performs the three operations of self-attention, convolution and nonlinearity in parallel, simply stacking this module outperforms all previous models including Transformer (Vasvani et al., 2017) on main NMT tasks under standard setting.

Key features:
  - Design a multi-branch schema evolving self attention and first successfully combine convolution and self-attention in one module for sequence tasks by the proposed shared projection,
  - SOTA on three main translation datasets, including WMT14 En-Fr, WMT14 En-De and IWSLT14 De-En,
  - Parallel learn sequence representations and thus have potential for acceleration.

Results:
1. Better than previous models on large NMT datasets; can scale to small datasets and base model setting. [Link](https://disk.pku.edu.cn:443/link/E53D94989506EE3E0AD2B9370C713E92)
2. The shared projection is key to combine conv and self-attn; generate better long sequences;potential for acceleration. [Link](https://disk.pku.edu.cn:443/link/E53D94989506EE3E0AD2B9370C713E92
)

| Task | size  | test (BLEU) |
| ---------- | ---:| ----:|
| IWSLT14 De-En | Base | 36.3 |
| WMT14 En-De |  Large  | 29.9 |
| WMT14 En-Fr |  Large | 43.5 |

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

**Installing from source**

To install MUSE from source and develop locally:
```
pip install --editable . --user
```

<!--# Pre-trained models and examples-->

We provide pre-trained models and detailed example training and
evaluation in [examples/parallel_multi-scale_attention(MUSE)/README.md](examples/parallel_multi-scale_attention(MUSE)/README.md).



<!--# License-->
<!--MIT-licensed.-->
<!--The license applies to the pre-trained models as well.-->
<!--We also provide an additional patent grant.-->

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

# Notes
The code is based on fairseq-0.6.2,
the main code can be seen in fairseq\models\combine_transformer.py(code for parallel representation learning) and fairseq\models\transformer_bm.py(means big matrix, code for acceleration), fairseq\modules\multihead_attention.py(code for combining convolution and self-attention)
