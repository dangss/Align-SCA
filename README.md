# Introduction
This repository contains the code used for our dissertation project: Improving Low-Resource Machine Translation

# Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)

After PyTorch is installed, you can install fairseq with:
```
pip install -r requirements.txt
python setup.py build develop
```

# Getting Started
To run SCA or Align-SCA model, you need to:
1. Train two language models for source language and target language.
2. Specify the path of the best checkpoint when trainig NMT system.

### Training of language model
An example of our script:
```
python train.py $DATA  --task language_modeling --arch $ARCH \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.0005 --min-lr 1e-09 \
  --dropout 0.15 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 4000  --tokens-per-sample 4000  --save-dir $SAVE 
```

You can change the language model architecture according to your dataset size.

### Training of NMT
When training with SCA or Align-SCA method, there are additional arguments that you need to include:
```
`--src-lm-path' is the path for source language model checkpoint.
`--tgt-lm-path` is the path for target language model
`--src-only` is flag to use SCA only on the encoder(default: False)
`--tgt-only` is flag for SCA only on the decoder(default:False)
`--sca-drop` is augmentation probability when use SCA
`--task` specifies if we use pre-train LM or not. If you use SCA or Align method, you do not need to change this. To train baseline or other model, you need to change it to 'translation_normal'.
`--arch` is the model you want to use (SCA: 'transformer_sca', SCA without BP: 'transformer_scabp', align: 'transformer_align', Align-SCA: 'transformer_ac')
```


An example of our script:
```

export CUDA_VISIBLE_DEVICES=0

src={src}
tgt={tgt}
DATA_PATH=fairseq/data-bin/{src}-{tgt}

SAVE_DIR=checkpoints/exp1
mkdir -p ${{SAVE_DIR}}

src_lm = fairseq/LM/{src}-{tgt}/{src}/checkpoint_best.pt
tgt_lm = fairseq/LM/{src}-{tgt}/{tgt}/checkpoint_best.pt

echo SAVE_DIR ${{SAVE_DIR}}


python train.py ${{DATA_PATH}} --arch transformer_sca --share-decoder-input-output-embed  \
--optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--lr 0.0005 --min-lr 1e-09 \
--dropout 0.15 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--encoder-layers 4 --decoder-layers 4 \
--encoder-embed-dim 256 --decoder-embed-dim 256 \
--encoder-attention-heads 4 --decoder-attention-heads 4 \
--encoder-ffn-embed-dim 384 --decoder-ffn-embed-dim 384 \
--max-tokens 4000 --max-update 70000 --src-lm-path $src_lm \ --tgt-lm-path $tgt_lm --save-dir ${{SAVE_DIR}} \
--sca-drop 0.1  --seed 94 
```

