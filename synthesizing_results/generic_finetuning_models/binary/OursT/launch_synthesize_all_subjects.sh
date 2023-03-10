#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export YOUR_PATH="/home/jingyitao/workspace/fNIRS_models/code_data_tufts"

for experiment in 64vs4 # 16vs4 64vs4 4vs4
do
    export experiment_dir="$YOUR_PATH/fNIRS-mental-workload-classifiers/experiments/generic_finetuning_models/OursT_1_3_2_1/binary/train_100/$experiment"
    
    echo "Current experiment_dir is $experiment_dir"
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < $YOUR_PATH/fNIRS-mental-workload-classifiers/synthesizing_results/generic_finetuning_models/binary/OursT/synthesize_all_subjects.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash $YOUR_PATH/fNIRS-mental-workload-classifiers/synthesizing_results/generic_finetuning_models/binary/OursT/synthesize_all_subjects.slurm
    fi
    
done