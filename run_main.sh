#!/bin/bash
DATA_PATH="./data/ETTh1.csv"
SAVE_PATH="./checkpoints/dino_patchtst"
N_GPUS=1
BATCH_SIZE=32 
python -m torch.distributed.run \
    --nproc_per_node=$N_GPUS \
    --master_port=29500 \
    main.py \
    --data_path "$DATA_PATH" \
    --output_dir "$SAVE_PATH" \
    --batch_size_per_gpu $BATCH_SIZE \
    --norm_last_layer True \
    --warmup_teacher_temp_epochs 10 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --clip_grad 3.0 \
    --num_patches 16 \
    --epochs 100 \
    --freeze_last_layer 2 \
    --lr 0.00005 \
    --c_in 7 \
    --warmup_epochs 10 \
    --min_lr 1e-6 \
    --optimizer "sgd" \
    --drop_path_rate 0.1 \
    --stride 12 \
    --transformation_group_size 6 \
