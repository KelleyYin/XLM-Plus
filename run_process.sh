#!/bin/bash


data=/data2/mmyin/XLM-experiments/data-bin/xlm-data-bin/zh-en-ldc-32k
vocab=$data/vocab
SRC=en
TGT=zh

for tst in train valid test; do
    for lang in $SRC $TGT; do
        echo $vocab.$lang
        echo $tst.$SRC-$TGT.$lang
        python preprocess.py $vocab.$lang $data/$tst.$SRC-$TGT.$lang
    done
done
