#!/bin/bash

data_bin=/data4/bjji/data/ldc/xlm_bin

export CUDA_VISIBLE_DEVICES=1,2,3,4
export NGPU=4

python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
    --exp_name Supervised_MT \
    --exp_id LDC_ch-en_no_share_vocab_label_smoothing_lr_0005_dropout03_share_langEmb_noAttnDrop\
    --dump_path ./checkpoints \
    --save_periodic 2 \
    --data_path $data_bin \
    --encoder_only False \
    --share_word_embeddings False \
    --use_lang_emb False \
    --sinusoidal_embeddings False \
    --share_all_embeddings False \
    --label_smoothing 0.1 \
    --lgs 'ch-en' \
    --clm_steps '' \
    --mlm_steps '' \
    --mt_steps 'ch-en' \
    --emb_dim 512 \
    --n_layers 6 \
    --n_heads 8 \
    --dropout 0.3 \
    --attention_dropout 0.1 \
    --gelu_activation False \
    --tokens_per_batch 7000 \
    --batch_size 32 \
    --bptt 256 \
    --master_port 3224 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0005 \
    --epoch_size 200000 \
    --eval_bleu True\
    --stopping_criterion 'valid_ch-en_mt_bleu,10' \
    --validation_metrics 'valid_ch-en_mt_bleu' 


