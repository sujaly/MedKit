#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
dataset="mimic_cxr"
model_type="base_cls"
annotation="data/mimic_cxr/annotation.json"
base_dir="./data/mimic_cxr/images"

version="train_base_cls"
savepath="save_supplement/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi


python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --model_type ${model_type} \
    --batch_size 2 \
    --val_batch_size 2 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 150 \
    --min_new_tokens 80 \
    --max_new_tokens 150 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 6 \
    --devices 4 \
    --max_epochs 4 \
    --limit_val_batches 0.1 \
    --val_check_interval 0.1 \
    --learning_rate 1e-5 \
    --num_sanity_val_steps 2 \
    --use_teacher true \
    2>&1 |tee -a ${savepath}/log.txt
