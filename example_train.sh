SEED=269902365
DATASET="plant-village-healthy-to-plant-village-rust"

CKPT_100K="iter_0100000"
CKPT_150K="iter_0150000"
CKPT_200K="iter_0200000"
CKPT_250K="iter_0250000"
CKPT_SMALLEST_VAL_FID="smallest_val_fid"

DISPLAY_FREQ=1000
EVAL_FREQ=10000
SAVE_FREQ=50000
ITERS=250000

BATCH_SIZE=1
LR=0.0001
NUM_BOTTLENECK_BLOCKS=9
NCE_WEIGHT=10.0
NCE_LAYERS="0,2,3,4,8"

python cli.py --phase train \
              --dataset $DATASET \
              --img_size 256 \
              --batch_size $BATCH_SIZE \
              --display_freq $DISPLAY_FREQ \
              --eval_freq $EVAL_FREQ \
              --save_freq $SAVE_FREQ \
              --iters $ITERS \
              --num_bottleneck_blocks $NUM_BOTTLENECK_BLOCKS \
              --nce_weight $NCE_WEIGHT \
              --nce_layers $NCE_LAYERS \
              --lr $LR \
              --iters $ITERS
