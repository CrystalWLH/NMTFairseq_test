#!/bin/bash

#evaluate experiment model

#only have source txt.
#CUDA_VISIBLE_DEVICES=3 python interactive.py mydata/data-bin/charTi_charZh --task translation --source-lang ti --target-lang zh --path checkpoints/charChar_exp1/checkpoint_best.pt --buffer-size 500 --batch-size 1 --beam 5 --remove-bpe --input ../../../data/CCMT2019/data_20191101/charTi_charZh/test-2019.ti

#have source txt and target txt.
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp5/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

CUDA_VISIBLE_DEVICES=0 python generate.py mydata/data-bin/nmt_seg_bin --source-lang ti --target-lang zh --seg-lang seg --task seg_translation --log-format simple --log-interval 10 --path checkpoints/exp1_2/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50
