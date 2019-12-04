#!/bin/bash

SRC=zh
TGT=en
src_file=/data2/mmyin/XLM-experiments/MT-data/zh-en-ldc-32k/test_set

model_file=./checkpoints/Supervised_MT/LDC_zh-en_not_share_vocab_label_smoothing_lr_0005_dropout03_share_langEmb_sinusoidal
mod
el=$model_file/best-valid_zh-en_mt_bleu.pth
# model=$model_file/periodic-36.pth

ref=/data2/mmyin/XLM-experiments/MT-data/zh-en-ldc-32k/test.zh-en.en

tst_sets="nist02 nist03 nist04 nist05 nist08"

export CUDA_VISIBLE_DEVICES=0
for tst in $tst_sets;do
    out_txt=$model_file/$tst.decoded.$TGT
    ref=$src_file/$tst.ref.
    src_txt=$src_file/$tst.bpe.in

    cat $src_txt | python translate.py --exp_name translation \
        --src_lang $SRC --tgt_lang $TGT \
        --model_path $model \
        --lenpen 1 \
        --beam_size 5 \
        --batch_size 60 \
        --output_path $out_txt.bpe
    sed -r 's/(@@ )|(@@ ?$)//g' $out_txt.bpe > $out_txt
#    perl ./src/evaluation/multi-bleu.perl $ref < $out_txt
done

for tst in $tst_sets;do
    ref=$src_file/$tst.ref.
    out_txt=$model_file/$tst.decoded.$TGT
    perl ./src/evaluation/multi-bleu.perl -lc $ref < $out_txt
done




















