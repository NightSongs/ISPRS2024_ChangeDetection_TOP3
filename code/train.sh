#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

save_path=logs

mkdir -p $save_path

python code/train_all_data.py 2>&1 | tee $save_path/$now.log