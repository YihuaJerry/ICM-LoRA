#!/bin/bash

SOURCE_PATH=../train_lora/model_checkpoints/xxxx #e.g dog-r=8
TARGET_PATH=../data/param_data/xxx  #e.g dog-r=8

python3 reformat_lora_param.py --source_path "$SOURCE_PATH" --target_path "$TARGET_PATH"

python3 normalizeLoraWeight_small --dataset_path "$TARGET_PATH"