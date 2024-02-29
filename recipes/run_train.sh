#!/bin/bash

# Copyright 2021 Hirokazu Kameoka
#
# Usage:
# ./run_train.sh [-g gpu] [-a arch_type] [-s stage] [-e exp_name]
# Options:
#     -g: GPU device#
#     -a: Architecture type ("conv" or "rnn")
#     -s: Stage to start (0 or 1)
#     -e: Experiment name (e.g., "exp1")

# Default values
db_dir="/data/hdd0/tianyi.tan/vcc2018/vcc2018_training"
dataset_name="vcc2018"
target_dir="/data/hdd0/tianyi.tan"
gpu=0
arch_type="conv"
start_stage=0
exp_name="conv_exp1"

while getopts "g:a:s:e:" opt; do
       case $opt in
              g ) gpu=$OPTARG;;
              a ) arch_type=$OPTARG;;
              s ) start_stage=$OPTARG;;
              e ) exp_name=$OPTARG;;
       esac
done

feat_dir="${target_dir}/dump/${dataset_name}/feat/train"
dconf_path="${target_dir}/dump/${dataset_name}/data_config.json"
stat_path="${target_dir}/dump/${dataset_name}/stat.pkl"
normfeat_dir="${target_dir}/dump/${dataset_name}/norm_feat/train"
model_dir="${target_dir}/model/${dataset_name}"
log_dir="${target_dir}/logs/${dataset_name}"

# Stage 0: Feature extraction
if [[ ${start_stage} -le 0 ]]; then
       python extract_features.py --src ${db_dir} --dst ${feat_dir} --conf ${dconf_path}
       python compute_statistics.py --src ${feat_dir} --stat ${stat_path}
       python normalize_features.py --src ${feat_dir} --dst ${normfeat_dir} --stat ${stat_path}
fi

# Stage 1: Model training
if [[ ${start_stage} -le 1 ]]; then
       python train.py -g ${gpu} \
              --data_rootdir ${normfeat_dir} \
              --model_rootdir ${model_dir} \
              --log_dir ${log_dir} \
              --arch_type ${arch_type} \
              --experiment_name ${exp_name} \
              ${cond}
fi
