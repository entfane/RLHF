#!/bin/bash

MODEL_NAME="HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET="argilla/ultrafeedback-binarized-preferences-cleaned"
DATASET_SPLIT="train"
PROMPT_COLUMN_NAME="prompt"
CHOSEN_COLUMN_NAME="chosen"
REJECTED_COLUMN_NAME="rejected"
EPOCHS=2
MINI_BATCH_SIZE=16
LOG_WANDB=True
MAX_LEN=4096
LR=0.00003

python main.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --dataset_split "$DATASET_SPLIT" \
    --prompt_column_name "$PROMPT_COLUMN_NAME" \
    --chosen_column_name "$CHOSEN_COLUMN_NAME" \
    --rejected_column_name "$REJECTED_COLUMN_NAME" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --max_len "$MAX_LEN" \
    --log_wandb "$LOG_WANDB" \
    --lr "$LR" 