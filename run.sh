#!/bin/bash

DATA_PATH="/Users/samin91/Desktop/Projects/ssl-semseg/Collections/Industrial_Burner_Flames_noAugmentation"
EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=1e-3
NUM_CLASSES=2  # grey scale images - foreground and background classes
PRETRAIN_TYPE="imagenet"  #["none", "imagenet", "simclr", "moco", "swav"]
PRETRAIN_PATH="/Users/samin91/Desktop/Projects/ssl-semseg/Weights"  #"Path to self-supervised VISSL weights (.pth)"
LOGGER_CSV="training_log.csv"
LOGGER_TENSOIRBOARD="runs"
IMG_PREDICTIONS="/Users/samin91/Desktop/Projects/ssl-semseg/img_predictions"
SUBSET_FRAC=0.01
FREEZE=True

python3 main.py \
    --data_root $DATA_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --num_classes $NUM_CLASSES \
    --pretrain_type $PRETRAIN_TYPE \
    --pretrain_path $PRETRAIN_PATH \
    --log_csv $LOGGER_CSV \
    --log_tb $LOGGER_TENSOIRBOARD \
    --img_prediction_path $IMG_PREDICTIONS \
    --subset_frac $SUBSET_FRAC \
    --freeze_backbone_fpn 
