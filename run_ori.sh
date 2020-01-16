#!/bin/bash

#deal data
#python preprocess.py --source-lang ti --seg-lang seg --target-lang zh --trainpref ../../../data/nmt_seg/train --validpref ../../../data/nmt_seg/valid --testpref ../../../data/nmt_seg/test-2017 --destdir mydata/data-bin/nmt_seg_bin --workers 5


#train transformer model
CUDA_VISIBLE_DEVICES=0 python train.py mydata/data-bin/wordTi_wordZh_v2 --task translation --arch lstm_wiseman_iwslt_de_en --save-dir checkpoints/test --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.1 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-tokens 4096


#evaluate experiment model
#CUDA_VISIBLE_DEVICES=7 python generate.py data/data-bin/charCCMT --task translation --log-format simple --log-interval 10 --path result/exp2/checkpoint_last.pt --batch-size 128 --beam 5 --remove-bpe --results-path temp

#python preprocess.py --source-lang ti --target-lang zh --trainpref ../../../data/CCMT2019/data_20191126/charTi_charZh/train --validpref ../../../data/CCMT2019/data_20191126/charTi_charZh/valid --testpref ../../../data/CCMT2019/data_20191126/charTi_charZh/test-2017 --destdir mydata/test --workers 5


