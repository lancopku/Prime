# MUSE: PARALLEL MULTI-SCALE ATTENTION FOR SEQUENCE TO SEQUENCE LEARNING (Zhao et al., 2019)

This page contains pointers to pre-trained models as well as instructions on how to train new models for the paper

## Preprocess data
See examples/translation

## Pretrained models

### Links of models and datasets
We  will provide pre-trained models for test.

Description | Dataset | Model | Test set(s)
---|---|---|---
Prime  | [IWSLT14 German-English](https://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz) | - | IWSLT14 test: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/iwslt14.de-en.test.tar.bz2)
Prime| [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [ende_prime_avg.pt](https://drive.google.com/open?id=1DoSYhBfAw07QStFj62uiDCDJKatBsBBc) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)
Prime| [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [enfr_prime_single_check.pt](https://drive.google.com/open?id=11BazzWdcSWyUtXXy1p_vHrhPowgd12nl) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)

### Evaluate the pretrained models
Place the models into directory of checkpoint and evaluate the models
#### Evaluate  En-Fr  on single checkpoint of the Prime
```sh
export CUDA_VISIBLE_DEVICES=0
python3 generate.py data-bin/wmt14_en_fr --path checkpoint/enfr_prime_single_check.pt --batch-size 64 --beam 5  \
--remove-bpe --lenpen 0.8 --gen-subset test --quiet > results/enfr_prime_single_check.txt
python3 generate.py data-bin/wmt14_en_fr --path checkpoint/enfr_prime_single_check.pt --batch-size 64 --beam 5  \
--remove-bpe --lenpen 0.8 --gen-subset valid --quiet > results/enfr_prime_single_check.txt
```
#### Evaluate  En-De  on the averaged checkpoint of the  Prime
```sh
export CUDA_VISIBLE_DEVICES=0
python3 generate.py data-bin/wmt16_en_de_bpe32k --path checkpoint/ende_prime_avg.pt --batch-size 128 --beam 4 --remove-bpe --lenpen 0.6 --gen-subset test --quiet > results/ende_prime_avg_test.txt
```

## Train  and evaluation
### Preprocessing the training datasets

Please follow the instructions in [`examples/translation/README.md`](../translation/README.md) to preprocess the data.

### Training and evaluation options:
For best BLEU results, lenpen, beam size  and checkpoints to average may need to be manually tuned.
### WMT14 En-Fr
Training and evaluating Prime on WMT16 En-Fr using cosine scheduler on one machine with 8 RTX 2080ti GPUs(11GB):
```sh
# Training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
save=enfr_prime_fp
blocks=12
dim=768
inner_dim=$((4*$dim))
cur_save=${save}
attn_dynamic_cat=1
attn_dynamic_type=2
kernel_size=0
python3 train.py data-bin/wmt14_en_fr \
  --arch transformer_vaswani_wmt_en_fr_big --share-all-embeddings --ddp-backend=no_c10d \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-shrink 1 --max-lr 0.0007 --lr 1e-7 --min-lr 1e-9 --lr-scheduler cosine --warmup-init-lr 1e-7 \
  --warmup-updates 10000 --t-mult 1 --lr-period-updates 70000 \
  --dropout 0.1  --attention-dropout 0.1 --weight-dropout 0.1 --input_dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 1280  --max-epoch 39 \
  --combine 1  --encoder-layers  ${blocks} --decoder-layers ${blocks} --encoder-embed-dim ${dim} --encoder-ffn-embed-dim ${inner_dim} --decoder-embed-dim ${dim} --decoder-ffn-embed-dim ${inner_dim}\
  --attn_dynamic_type ${attn_dynamic_type} --kernel_size ${kernel_size} --attn_dynamic_cat ${attn_dynamic_cat}   --attn_wide_kernels [3,15]  --dynamic_gate 1  \
  --update-freq 76 --fp16  --tensorboard-logdir checkpoint/${cur_save}  --log-format json --save-dir checkpoint/${cur_save} 2>&1 | tee checkpoint/${cur_save}.txt


# Evaluation
export CUDA_VISIBLE_DEVICES=0
python3 generate.py data-bin/wmt14_en_fr --path checkpoint/${cur_save}/checkpoint_best.pt --batch-size 64 --beam 5  \
--remove-bpe --lenpen 0.8 --gen-subset test --quiet > results/${cur_save}.txt
python3 generate.py data-bin/wmt14_en_fr --path checkpoint/${cur_save}/checkpoint_best.pt --batch-size 64 --beam 5  \
--remove-bpe --lenpen 0.8 --gen-subset valid --quiet > results/${cur_save}_valid.txt
```

Training and evaluating Prime on WMT16 En-Fr using cosine scheduler on one machine with 4 RTX TITAN GPUs(24GB):
```sh
# Training
export CUDA_VISIBLE_DEVICES=0,1,2,3
save=enfr_prime_fp
blocks=12
dim=768
inner_dim=$((4*$dim))
cur_save=${save}
attn_dynamic_cat=1
attn_dynamic_type=2
kernel_size=0
python3 train.py data-bin/wmt14_en_fr \
  --arch transformer_vaswani_wmt_en_fr_big --share-all-embeddings --ddp-backend=no_c10d \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-shrink 1 --max-lr 0.0007 --lr 1e-7 --min-lr 1e-9 --lr-scheduler cosine --warmup-init-lr 1e-7 \
  --warmup-updates 10000 --t-mult 1 --lr-period-updates 70000 \
  --dropout 0.1  --attention-dropout 0.1 --weight-dropout 0.1 --input_dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 5120  --max-epoch 39  \
  --combine 1 --encoder-layers  ${blocks} --decoder-layers ${blocks} --encoder-embed-dim ${dim} --encoder-ffn-embed-dim ${inner_dim} --decoder-embed-dim ${dim} --decoder-ffn-embed-dim ${inner_dim}\
  --attn_dynamic_type ${attn_dynamic_type} --kernel_size ${kernel_size} --attn_dynamic_cat ${attn_dynamic_cat}   --attn_wide_kernels [3,15]  --dynamic_gate 1  \
  --update-freq 32 --fp16  --tensorboard-logdir checkpoint/${cur_save}  --log-format json --save-dir checkpoint/${cur_save} 2>&1 | tee checkpoint/${cur_save}.txt


# Evaluation
export CUDA_VISIBLE_DEVICES=0
python3 generate.py data-bin/wmt14_en_fr --path checkpoint/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5  \
--remove-bpe --lenpen 0.8 --gen-subset valid --quiet > results/${cur_save}_valid.txt
python3 generate.py data-bin/wmt14_en_fr --path checkpoint/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5  \
--remove-bpe --lenpen 0.8 --gen-subset test --quiet > results/${cur_save}.txt
```

### WMT16 En-De

Training and evaluating Prime on WMT16 En-De using cosine scheduler on one machine with 4 Titan RTX gpus:
```sh
# Training
export CUDA_VISIBLE_DEVICES=0,1,2,3
save=ende_prime_fp
blocks=12
dim=768
inner_dim=$((4*$dim))
cur_save=${save}
attn_dynamic_cat=1
attn_dynamic_type=2
kernel_size=0
python3 train.py data-bin/wmt16_en_de_bpe32k \
  --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --ddp-backend=no_c10d \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --ddp-backend=no_c10d --max-tokens  3584 \
  --lr-scheduler cosine --warmup-updates 10000 --lr-shrink 1 --max-lr 0.001 --lr 1e-7 --min-lr 1e-9 --warmup-init-lr 1e-07 --t-mult 1 --lr-period-updates 20000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 --input_dropout 0.1 \
  --fp16 \
  --combine 1 --encoder-layers  ${blocks} --decoder-layers ${blocks} --encoder-embed-dim ${dim} --encoder-ffn-embed-dim ${inner_dim} --decoder-embed-dim ${dim} --decoder-ffn-embed-dim ${inner_dim}\
  --attn_dynamic_type ${attn_dynamic_type} --kernel_size ${kernel_size} --attn_dynamic_cat ${attn_dynamic_cat}   --attn_wide_kernels [3,15] --dynamic_gate 1 \
  --update-freq 32 --tensorboard-logdir checkpoint/${cur_save}  --log-format json --save-dir checkpoint/${cur_save} 2>&1 | tee checkpoint/${cur_save}.txt

# Evaluation
python3 generate.py data-bin/wmt16_en_de_bpe32k --path checkpoint/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe --lenpen 0.6 --gen-subset test > results/${cur_save}_checkpoint_best.txt
# Average, generate and compound spliting
```

Training and evaluating the Prime-simple on WMT16 En-De using invert-sqrt scheduler on one machine with 4 Titan RTX gpus:
```sh
# Training
export CUDA_VISIBLE_DEVICES=0,1,2,3
save=ende_prime_simple_fp
blocks=12
dim=768
inner_dim=$((4*$dim))
lr=0.001
cur_save=${save}
python3 train.py data-bin/wmt16_en_de_bpe32k \
  --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --ddp-backend=no_c10d \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr ${lr} --min-lr 1e-09 \
  --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 3584  --max-epoch 75 \
  --combine 1 --encoder-layers  ${blocks} --decoder-layers ${blocks} --encoder-embed-dim ${dim} --encoder-ffn-embed-dim ${inner_dim} --decoder-embed-dim ${dim} --decoder-ffn-embed-dim ${inner_dim}\
  --fp16 --update-freq 40 --tensorboard-logdir checkpoint/${cur_save}  --log-format json --save-dir checkpoint/${cur_save} 2>&1 | tee checkpoint/${cur_save}.txt


# Evaluation
python3 generate.py data-bin/wmt16_en_de_bpe32k --path checkpoint/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe --lenpen 0.6 --gen-subset test > results/${cur_save}_checkpoint_best.txt
# Average, generate and compound spliting
```


### IWSLT14 De-En
Training and evaluating Prime on a GPU:
```sh
# Training
export CUDA_VISIBLE_DEVICES=0
save=deen_prime_fp
blocks=12
dim=384
inner_dim=$((2*$dim))
attn_dynamic_cat=1
attn_dynamic_type=2
kernel_size=0
seed=0
cur_save=${save}
python3 train.py data-bin/iwslt14.tokenized.de-en -a transformer_iwslt_de_en --optimizer adam --lr 0.001 -s de -t en --label-smoothing 0.1 --dropout 0.4 --max-tokens 4000 \
      --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
      --criterion label_smoothed_cross_entropy --max-update 20000 \
      --warmup-updates 4000 --warmup-init-lr '1e-07'  --update-freq 4 --keep-last-epochs 30 \
      --adam-betas '(0.9, 0.98)' --save-dir checkpoint/${cur_save}  \
      --attn_dynamic_type ${attn_dynamic_type} --kernel_size ${kernel_size} --attn_dynamic_cat ${attn_dynamic_cat}   --attn_wide_kernels [3,15] --fp16 \
      --combine 1  --encoder-layers  ${blocks} --decoder-layers ${blocks}  \
      --encoder-embed-dim ${dim} --encoder-ffn-embed-dim ${inner_dim} --decoder-embed-dim ${dim} --decoder-ffn-embed-dim ${inner_dim} --seed ${seed} \
      --log-format json --tensorboard-logdir checkpoint/${cur_save}  2>&1 | tee checkpoint/${cur_save}.txt
# Evaluation
python3 generate.py data-bin/iwslt14.tokenized.de-en --path checkpoint/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_checkpoint_best.txt
```

Training and evaluating Prime-simple on a GPU:
```sh
# Training
export CUDA_VISIBLE_DEVICES=0
save=deen_prime_simple_fp
blocks=12
dim=384
inner_dim=$((2*$dim))
cur_save=${save}
python3 train.py data-bin/iwslt14.tokenized.de-en -a transformer_iwslt_de_en --optimizer adam --lr 0.001 -s de -t en --label-smoothing 0.1 --dropout 0.4 --max-tokens 4000 \
      --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
      --criterion label_smoothed_cross_entropy --max-update 20000 \
      --warmup-updates 4000 --warmup-init-lr '1e-07'  --update-freq 4 --keep-last-epochs 30 \
      --adam-betas '(0.9, 0.98)' --save-dir checkpoint/${cur_save}  --fp16 \
      --combine 1  --encoder-layers  ${blocks} --decoder-layers ${blocks}  \
      --encoder-embed-dim ${dim} --encoder-ffn-embed-dim ${inner_dim} --decoder-embed-dim ${dim} --decoder-ffn-embed-dim ${inner_dim} --seed ${seed} \
      --log-format json --tensorboard-logdir checkpoint/${cur_save}  2>&1 | tee checkpoint/${cur_save}.txt
# Evaluation
python3 generate.py data-bin/iwslt14.tokenized.de-en --path checkpoint/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_checkpoint_best.txt
```
### Inference Speed for simple  module on IWSLT de-en  with single RTX TITAN
```sh
save=deen_prime_simple_fp
export CUDA_VISIBLE_DEVICES=0
blocks=6
dim=512
inner_dim=$((2*$dim))
results_name=bm_speed
for seed in 1
do
    cur_save=${save}_bm_s${seed}
    python3 train.py data-bin/iwslt14.tokenized.de-en -a transformer_iwslt_de_en --optimizer adam --lr 0.001 -s de -t en --label-smoothing 0.1 --dropout 0.4 --max-tokens 4000 \
     --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
     --criterion label_smoothed_cross_entropy --max-updates 20000 \
     --warmup-updates 4000 --warmup-init-lr '1e-07'  --update-freq 4 --keep-last-epochs 30 \
     --adam-betas '(0.9, 0.98)' --save-dir checkpoint/${cur_save} --seed ${seed} --fp16 \
     --bm 1 --bm_in_a 3 --bm_out_a 0 \
     --encoder-layers  ${layers} --decoder-layers ${layers} \
     --encoder-embed-dim ${dim} --encoder-ffn-embed-dim ${inner_dim} --decoder-embed-dim ${dim} --decoder-ffn-embed-dim ${inner_dim} \
     --log-format json --tensorboard-logdir checkpoint/${cur_save}  2>&1 | tee checkpoint/${cur_save}.txt
    python3 average_checkpoints.py --inputs checkpoint/$cur_save  --num-epoch-checkpoints 10 --output checkpoint/$cur_save/avg_.pt
    python3 generate.py data-bin/iwslt14.tokenized.de-en --path checkpoint/$cur_save/avg_${i}.pt --batch-size 1 --beam 5 --remove-bpe --quiet  > results/${results_name}/${cur_save}_test.txt
done
```
