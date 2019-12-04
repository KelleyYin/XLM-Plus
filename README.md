# XLM-Plus
## Advantages of XLM
[XLM](https://github.com/facebookresearch/XLM) is supported by Fackbook. It is a very good project for neural machine translation reseachers.    
Because this project have implemented auto-encoder, various LMs, unsupervised NMT and supervised NMT.  In addition, XLM is implemented by Pytorch, and also supports multi-GPU and multi-node training.
Therefore, it is very convenience that the researcher implement their own experiments on this project.

## XLM-Plus
Although XLM brings great convenience, it has the following disadvantages.
* Not support independent vocabulary training for different language 
* [The baseline of supervised NMT is lower than fairseq and tensor2tensor](https://github.com/facebookresearch/XLM/issues/32). 

According to the above problems, I added some functions to the source code to alleviate the disadvantages.
* Support independent vocab for different languages between encoder and decoder
* Add label smoothing function for criterion
* Share a same word embedding table for similar languages(eg., en-de) between encoder and decoder
* Specific scripts for training and decoding process

Traning script as follow.

```
data_bin=/data2/mmyin/XLM-experiments/data-bin/xlm-data-bin/zh-en-ldc-32k

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=4

python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
    --exp_name Supervised_MT \
    --exp_id LDC_zh-en_not_share_vocab_label_smoothing_update\
    --dump_path ./checkpoints \
    --save_periodic 2 \
    --data_path $data_bin \
    --encoder_only False \
    --share_word_embeddings False \
    --share_lang_embeddings False \
    --share_all_embeddings False \
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

```

Notes:    
`--share_word_embeddings`: sharing a same word embedding tabel for different languages      
`--share_lang_embeddings`: sharing a same language embedding tabel between encoder and decoder    
`--share_all_embeddings`: conbining the above two methods    
`--label_smoothing`: using label smoothing criterion


Decoding script as follow.

```
SRC=zh
TGT=en
src_file=/data2/mmyin/XLM-experiments/MT-data/zh-en-ldc-32k/test_set

model_file=/data2/mmyin/XLM-experiments/XLM-update/checkpoints/Supervised_MT/LDC_zh-en_not_share_vocab_label_smoothing
#model=$model_file/best-valid_zh-en_mt_bleu.pth
model=$model_file/checkpoint.pth

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
        --batch_size 32 \
        --output_path $out_txt.bpe
    sed -r 's/(@@ )|(@@ ?$)//g' $out_txt.bpe > $out_txt
#    perl ./src/evaluation/multi-bleu.perl $ref < $out_txt
done

for tst in $tst_sets;do
    ref=$src_file/$tst.ref.
    out_txt=$model_file/$tst.decoded.$TGT
    perl ./src/evaluation/multi-bleu.perl -lc $ref < $out_txt
done
```


 

## Experiment Test

nist02 | nist03 | nist04 | nist05 | nist08 | avg | Note
---|---|---|---|---|--- | ----
47.16 | 46.16 | 47.09 | 46.33 | 38.11 | 44.97 | fairseq-baseline
42.90 | 40.73 | 42.95 | 41.58 | 33.57 | 40.35 | share_vocab
44.41 | 42.74 | 44.54 | 43.52 | 35.45 | 42.13 | independent_vocab
44.82 | 43.49 | 44.63 | 43.10 | 35.16 | 42.24 | labelSmoothing
45.63 | 43.99 | 45.86 | 45.00 | 36.30 | 43.36 | labelSmoothing+dp03
45.86 | 45.13 | 46.66 | 45.48 | 36.96 | 44.02 | labelsmoothing+dp03+NoLangEmb
45.36 | 44.19 | 46.41 | 45.86 | 37.07 | 43.78 | label+dp03+shareLangEmb
46.19 | 43.98 | 46.43 | 45.43 | 36.55 | 43.72 | label+dp03+shareLangEmb+Drop
45.17 | 44.27 | 46.38 | 44.55 | 36.81 | 43.44 | label+dp03+shareLangEmb+Sim



