#!/bin/bash

# Copyright 2021 Hirokazu Kameoka
# 
# Usage:
# ./run_test_arctic_4spk.sh [-g gpu] [-e exp_name] [-c checkpoint] [-v vocoder_type]
# Options:
#     -g: GPU device#  
#     -e: Experiment name (e.g., "conv_exp1")
#     -c: Model checkpoint to load (0 indicates the newest model)
#     -v: Vocoder type ("hifigan.v1" or "parallel_wavegan.v1")

db_dir="/data/hdd0/tianyi.tan/vcc2018/vcc2018_reference"
dataset_name="vcc2018"
target_dir="/data/hdd0/tianyi.tan"
gpu=0
checkpoint=0
vocoder_type="parallel_wavegan.v1"

while getopts "g:e:c:v:" opt; do
       case $opt in
              g ) gpu=$OPTARG;;
              e ) exp_name=$OPTARG;;
			  c ) checkpoint=$OPTARG;;
			  v ) vocoder_type=$OPTARG;;
       esac
done

# If the -v option is abbreviated...
case ${vocoder_type} in
	"pwg" ) vocoder_type="parallel_wavegan.v1";;
	"hfg" ) vocoder_type="hifigan.v1";;
esac

echo "Experiment name: ${exp_name}, Vocoder: ${vocoder_type}"

dconf_path="${target_dir}/dump/${dataset_name}/data_config.json"
stat_path="${target_dir}/dump/${dataset_name}/stat.pkl"
out_dir="${target_dir}/out/${dataset_name}"
model_dir="${target_dir}/model/${dataset_name}"
# vocoder_dir="${target_dir}/pwg/egs/arctic_4spk_flen64ms_fshift8ms/voc1"
# vocoder_dir="${target_dir}/pwg/egs/arctic_5spk_flen64ms_fshift8ms/voc1"
# vocoder_dir="${target_dir}/pwg/egs/ATR_all_flen64ms_fshift8ms/voc1"
vocoder_dir="${target_dir}/pwg/egs/seiyu_flen64ms_fshift8ms/voc1"

python convert.py -g ${gpu} \
	--input ${db_dir} \
	--dataconf ${dconf_path} \
	--stat ${stat_path} \
	--out ${out_dir} \
	--model_rootdir ${model_dir} \
	--experiment_name ${exp_name} \
	--vocoder ${vocoder_type} \
	--voc_dir ${vocoder_dir} \
	--checkpoint ${checkpoint}