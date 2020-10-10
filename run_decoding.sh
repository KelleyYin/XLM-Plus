# #!/bin/bash

SRC=ch
TGT=en
src_file=/data4/bjji/data/ldc

model_file=checkpoints/Supervised_MT/LDC_ch-en_no_share_vocab_label_smoothing_lr_0005_dropout03_share_langEmb_noAttnDrop/
model=$model_file/best-valid_ch-en_mt_bleu.pth

ref=/data4/bjji/data/ldc

tst_sets="nist02 nist03 nist04 nist05 nist08"

# export CUDA_VISIBLE_DEVICES=5
# for tst in $tst_sets;do
#     out_txt=$model_file/$tst.decoded.$TGT
#     ref=$src_file/$tst.ref.
#     src_txt=$src_file/$tst.bpe.in

#     cat $src_txt | python translate.py --exp_name translation \
#         --src_lang $SRC --tgt_lang $TGT \
#         --model_path $model \
#         --lenpen 1 \
#         --beam_size 5 \
#         --batch_size 60 \
#         --output_path $out_txt.bpe
#     sed -r 's/(@@ )|(@@ ?$)//g' $out_txt.bpe > $out_txt
# done

for tst in $tst_sets;do
    ref=$src_file/$tst.ref.
    out_txt=$model_file/$tst.decoded.$TGT
    perl ./src/evaluation/multi-bleu.perl -lc $ref < $out_txt
done











