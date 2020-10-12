#!/bin/bash

data_bin=/data4/bjji/data/eurporal

export CUDA_VISIBLE_DEVICES=1,2,3,4
export NGPU=4

python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
    --exp_name Multilingual_MT \
    --exp_id Eurporal_en-de-fr-es\
    --dump_path ./checkpoints \
    --save_periodic 2 \
    --data_path $data_bin \
    --encoder_only False \
    --share_word_embeddings True \
    --use_lang_emb False \
    --sinusoidal_embeddings False \
    --share_all_embeddings False \
    --label_smoothing 0.1 \
    --lgs 'en-es-de-fr' \
    --clm_steps '' \
    --mlm_steps '' \
    --mt_steps 'en-es,en-de,en-fr,es-en,es-de,es-fr,de-en,de-es,de-fr,fr-en,fr-es,fr-de' \
    --zero_shot es-de es-fr fr-de fr-es de-fr de-es \
    --emb_dim 512 \
    --n_layers 6 \
    --n_heads 8 \
    --dropout 0.1 \
    --attention_dropout 0 \
    --gelu_activation False \
    --tokens_per_batch 2500 \
    --batch_size 32 \
    --bptt 256 \
    --master_port 3224 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0005 \
    --epoch_size 200000 \
    --eval_bleu True\
    --stopping_criterion 'valid_en-fr_mt_bleu,100' \
    --validation_metrics 'valid_en-fr_mt_bleu' \
    --mnmt True  --eval_num 100


