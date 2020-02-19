#!/bin/bash

#deal data
#python preprocess.py --source-lang ti --seg-lang seg --target-lang zh --trainpref ../../../data/nmt_seg/train --validpref ../../../data/nmt_seg/valid --testpref ../../../data/nmt_seg/test-2017 --destdir mydata/data-bin/nmt_seg_bin --workers 5

#python preprocess.py --source-lang zh --seg-lang seg --target-lang en --trainpref ../../../data/nmt_seg_ZhEn/train --validpref ../../../data/nmt_seg_ZhEn/valid --testpref ../../../data/nmt_seg_ZhEn/test --destdir mydata/data-bin/nmt_seg_ZhEn_bin --workers 5 --nwordssrc 6000 --nwordsseg 50000 --nwordstgt 50000

#train transformer model
#CUDA_VISIBLE_DEVICES=0 python train.py mydata/data-bin/nmt_seg_bin --source-lang ti --target-lang zh --seg-lang seg --task seg_translation --arch seg_nmt_ctc_lstm --save-dir checkpoints/exp1 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4  --criterion ctc_nmt_loss --max-tokens 4096
#CUDA_VISIBLE_DEVICES=0 python train.py mydata/data-bin/nmt_seg_bin --source-lang ti --target-lang zh --seg-lang seg --task seg_translation --arch seg_nmt_ctc_lstm --save-dir checkpoints/exp1_1 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4  --criterion ctc_nmt_loss --max-tokens 4096 --restore-file checkpoints/exp1/checkpoint_best.pt --ctc-weight 0.1 --nmt-weight 0.9 --reset-meters
#CUDA_VISIBLE_DEVICES=0 python train.py mydata/data-bin/nmt_seg_bin --source-lang ti --target-lang zh --seg-lang seg --task seg_translation --arch seg_nmt_ctc_lstm --save-dir checkpoints/exp1_2 --log-interval 300 --no-progress-bar  --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4  --criterion ctc_nmt_loss --max-tokens 4096 --restore-file checkpoints/exp1/checkpoint_best.pt --ctc-weight 0.5 --nmt-weight 0.5 --reset-meters
#CUDA_VISIBLE_DEVICES=0 python train.py mydata/data-bin/nmt_seg_bin --source-lang ti --target-lang zh --seg-lang seg --task seg_translation --arch seg_nmt_ctc_lstm --save-dir checkpoints/exp1_3 --log-interval 300 --no-progress-bar  --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4  --criterion ctc_nmt_loss --max-tokens 4096 --restore-file checkpoints/exp1/checkpoint_best.pt --ctc-weight 0.05 --nmt-weight 0.95 --reset-meters --save-interval 20
#CUDA_VISIBLE_DEVICES=0 python train.py mydata/data-bin/nmt_seg_bin --source-lang ti --target-lang zh --seg-lang seg --task seg_translation --arch seg_nmt_ctc_lstm --save-dir checkpoints/exp1_4 --log-interval 300 --no-progress-bar  --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4  --criterion ctc_nmt_loss --max-tokens 4096 --restore-file checkpoints/exp1/checkpoint_best.pt --ctc-weight 0 --nmt-weight 1.0 --reset-meters --save-interval 20
#CUDA_VISIBLE_DEVICES=1 python train.py mydata/data-bin/nmt_seg_bin --source-lang ti --target-lang zh --seg-lang seg --task seg_translation --arch my_ctc_nmt_lstm --save-dir checkpoints/exp2 --log-interval 300 --no-progress-bar --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4  --criterion ctc_nmt_loss --max-tokens 4096 --ctc-weight 0 --nmt-weight 1
#CUDA_VISIBLE_DEVICES=1 python train.py mydata/data-bin/nmt_seg_bin --source-lang ti --target-lang zh --seg-lang seg --task seg_translation --arch my_ctc_nmt_lstm --save-dir checkpoints/exp3 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4  --criterion ctc_nmt_loss --max-tokens 4096 --ctc-weight 0 --nmt-weight 1

#---------------bidirection------------
#char-word的baseline
#CUDA_VISIBLE_DEVICES=1 python train.py mydata/data-bin/nmt_seg_bin --source-lang ti --target-lang zh --seg-lang seg --task seg_translation --arch my_ctc_nmt_lstm --save-dir checkpoints/exp4 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --optimizer adam --lr 0.0005 --lr-shrink 0.5 --criterion ctc_nmt_loss --max-tokens 4096 --ctc-weight 0 --nmt-weight 1 --dropout 0.3 --nmt-encoder-hidden-size 1024
#CUDA_VISIBLE_DEVICES=0 python train.py mydata/data-bin/nmt_seg_ZhEn_bin --source-lang zh --target-lang en --seg-lang seg --task seg_translation --arch my_ctc_nmt_lstm --save-dir checkpoints/exp4_zhen --log-interval 1000 --no-progress-bar --no-epoch-checkpoints --log-format simple --optimizer adam --lr 0.0005 --lr-shrink 0.5 --criterion ctc_nmt_loss --max-tokens 2048 --ctc-weight 0 --nmt-weight 1 --dropout 0.3 --nmt-encoder-hidden-size 1024

#char-word的ctc训练
#CUDA_VISIBLE_DEVICES=1 python train.py mydata/data-bin/nmt_seg_bin --source-lang ti --target-lang zh --seg-lang seg --task seg_translation --arch my_ctc_nmt_lstm --save-dir checkpoints/exp5 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --optimizer adam --lr 0.0005 --lr-shrink 0.5 --criterion ctc_nmt_loss --max-tokens 2048 --ctc-weight 1 --nmt-weight 0 --dropout 0.3 
#CUDA_VISIBLE_DEVICES=1 python train.py mydata/data-bin/nmt_seg_bin --source-lang ti --target-lang zh --seg-lang seg --task seg_translation --arch my_ctc_nmt_lstm --save-dir checkpoints/exp5_1 --log-interval 300 --no-progress-bar  --log-format simple --optimizer adam  --lr 0.0005 --lr-shrink 0.5  --criterion ctc_nmt_loss --max-tokens 2048 --restore-file checkpoints/exp5/checkpoint_best.pt --ctc-weight 0.1 --nmt-weight 0.9 --reset-meters --save-interval 20

CUDA_VISIBLE_DEVICES=1 python train.py mydata/data-bin/nmt_seg_ZhEn_bin --source-lang zh --target-lang en --seg-lang seg --task seg_translation --arch my_ctc_nmt_lstm --save-dir checkpoints/exp5_zhen --log-interval 1000 --no-progress-bar --no-epoch-checkpoints --log-format simple --optimizer adam --lr 0.0005 --lr-shrink 0.5 --criterion ctc_nmt_loss --max-tokens 2048 --ctc-weight 1 --nmt-weight 0 --dropout 0.3
