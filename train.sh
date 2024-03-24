#!/usr/bin/env bash
#!/bin/bash
#source /etc/profile
python_path=/root/miniconda3/envs/py36/bin/
root_path=/run/zwj/albert-chinese-ner

cd $root_path
export PYTHON=${python_path}/python
export CUDA_VISIBLE_DEVICES=7

output_dir=output_$(date +"%m%d%H%M")
mkdir ${output_dir}
echo "output:${output_dir}"




${PYTHON} albert_ner.py --task_name ner --do_train true --do_eval true --data_dir data --vocab_file ./albert_config/vocab.txt --bert_config_file ./albert_base_zh/albert_config_base.json --max_seq_length 512 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 10 --output_dir ${output_dir}

echo "done"