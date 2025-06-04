#!/bin/bash
dataset="mimic_cxr"
model_type='base_cls_ts'
annotation="data/mimic_cxr/annotation.json"
base_dir="data/mimic_cxr/images"
delta_file=""

version="base_cls"
savepath="./save_supplement/$dataset/$version"


if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train.py \
    --test \
    --dataset ${dataset} \
    --model_type ${model_type} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 6 \
    --max_length 180 \
    --min_new_tokens 80 \
    --max_new_tokens 180 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --num_workers 8 \
    --devices 4 \
    2>&1 |tee -a ${savepath}/log.txt
