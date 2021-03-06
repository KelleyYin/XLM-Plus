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
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0005 \
    --epoch_size 200000 \
    --eval_bleu True\
    --stopping_criterion 'valid_ch-en_mt_bleu,10' \
    --validation_metrics 'valid_ch-en_mt_bleu' 

```

Notes:    
`--share_word_embeddings`: sharing the same word embedding for different languages      
`--share_all_embeddings`: combining the above two methods    
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
48.16 | 46.73 | 47.92 | 48.09 | 38.65 | 45.91 | labelsmoothing+dp03+NoLangEmb



