#!/bin/bash


data=/data4/bjji/data/ldc
vocab=$data/vocab
SRC=en
TGT=ch

for tst in train valid test; do
    for lang in $SRC $TGT; do
        python preprocess.py $vocab.$lang $data/$tst.bpe.$lang
    done
done
