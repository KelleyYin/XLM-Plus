#!/bin/bash

data_bin=/data2/mmyin/XLM-experiments/data-bin/xlm-data-bin/zh-en-ldc-32k

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=4

python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
    --exp_name Supervised_MT \
    --exp_id LDC_zh-en_not_share_vocab_label_smoothing_lr_0001\
    --dump_path ./checkpoints \
    --save_periodic 2 \
    --data_path $data_bin \
    --encoder_only False \
    --share_word_embeddings False \
    --label_smoothing 0.1 \
    --lgs 'en-zh' \
    --clm_steps '' \
    --mlm_steps '' \
    --mt_steps 'zh-en' \
    --emb_dim 512 \
    --n_layers 6 \
    --n_heads 8 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --gelu_activation True \
    --tokens_per_batch 2000 \
    --batch_size 32 \
    --bptt 256 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size 200000 \
    --eval_bleu True\
    --stopping_criterion 'valid_zh-en_mt_bleu,10' \
    --validation_metrics 'valid_zh-en_mt_bleu'


