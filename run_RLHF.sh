#!/bin/bash

MODEL_NAME="HuggingFaceTB/SmolLM2-135M-Instruct"
REWARD_MODEL_NAME="Skywork/Skywork-Reward-V2-Qwen3-0.6B"
DATASET="TuringEnterprises/Turing-Open-Reasoning"
DATASET_SPLIT="train"
PROMPT_COLUMN="question"
BETA=0.01
ITERATIONS=3
BATCH_SAMPLING_PERCENTAGE=1
MINI_BATCH_SIZE=2
EPOCHS=2
MAX_NEW_TOKENS=128
GAMMA=0.98
LAMBDA=0.9
EPSILON=0.1
VALUE_LOSS_COEF=0.5
ENTROPY_LOSS_COEF=0.01
FREQUENCY_OF_COMPLETION_LOGGING=1
LOG_WANDB=True
LR=0.000002

python src/main.py \
    --model_name "$MODEL_NAME" \
    --reward_model_name "$REWARD_MODEL_NAME" \
    --dataset "$DATASET" \
    --dataset_split "$DATASET_SPLIT" \
    --prompt_column_name "$PROMPT_COLUMN" \
    --iterations "$ITERATIONS" \
    --batch_sampling_percentage "$BATCH_SAMPLING_PERCENTAGE" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --gamma "$GAMMA" \
    --lmbda "$LAMBDA" \
    --epsilon "$EPSILON" \
    --value_loss_coef "$VALUE_LOSS_COEF" \
    --entropy_loss_coef "$ENTROPY_LOSS_COEF" \
    --frequency_of_completion_logging "$FREQUENCY_OF_COMPLETION_LOGGING" \
    --beta "$BETA" \
    --log_wandb "$LOG_WANDB" \
    --lr "$LR"