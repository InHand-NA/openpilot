#!/bin/bash

TRAIN_DATASET="data/dual_camera_train/Town04_Opt"

python tools/dashcam/train/train.py \
    --data-dirs $TRAIN_DATASET \
    --dual-camera \
    --output-dir checkpoints/dual_v2 \
    --epochs 100 --batch-size 16 --early-stop 20