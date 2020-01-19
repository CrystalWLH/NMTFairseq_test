#!/bin/bash

#deal data
#python preprocess.py --source-lang ti --seg-lang seg --target-lang zh --trainpref ../../../data/nmt_seg/train --validpref ../../../data/nmt_seg/valid --testpref ../../../data/nmt_seg/test-2017 --destdir mydata/data-bin/nmt_seg_bin --workers 5


#train transformer model
CUDA_VISIBLE_DEVICES=0 python train.py mydata/data-bin/nmt_seg_bin --source-lang ti --target-lang zh --seg-lang seg --task seg_translation --arch seg_nmt_ctc_lstm --save-dir checkpoints/test --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4  --criterion ctc_nmt_loss --max-tokens 4096


#evaluate experiment model
#CUDA_VISIBLE_DEVICES=7 python generate.py data/data-bin/charCCMT --task translation --log-format simple --log-interval 10 --path result/exp2/checkpoint_last.pt --batch-size 128 --beam 5 --remove-bpe --results-path temp

#python preprocess.py --source-lang ti --target-lang zh --trainpref ../../../data/CCMT2019/data_20191126/charTi_charZh/train --validpref ../../../data/CCMT2019/data_20191126/charTi_charZh/valid --testpref ../../../data/CCMT2019/data_20191126/charTi_charZh/test-2017 --destdir mydata/test --workers 5


