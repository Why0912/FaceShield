#!/usr/bin/env bash
set -euo pipefail

MODEL_TYPE=${MODEL_TYPE:-phi-3}
BASE_LLM=${BASE_LLM:-/path/to/Phi-3-mini-4k-instruct}
VISION_TOWER=${VISION_TOWER:-/path/to/siglip-so400m-patch14-384}
TRAIN_JSON=${TRAIN_JSON:-/path/to/faceshield_pretrain.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/path/to/images}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints-pretrain/faceshield-pretrain}

mkdir -p "${OUTPUT_DIR}"

deepspeed bunny/train/train.py \
  --deepspeed ./script/deepspeed/zero2.json \
  --model_name_or_path "${BASE_LLM}" \
  --model_type "${MODEL_TYPE}" \
  --version phi3 \
  --data_path "${TRAIN_JSON}" \
  --image_folder "${IMAGE_FOLDER}" \
  --vision_tower "${VISION_TOWER}" \
  --mm_projector_type mlp2x_gelu \
  --tune_mm_mlp_adapter True \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 5e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none 2>&1 | tee "${OUTPUT_DIR}/log.txt"
