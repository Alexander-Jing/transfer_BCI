#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either "list" or "submit" or "run_here"

if [[ -z $1 ]]; then
    ACTION_NAME="list"
else
    ACTION_NAME=$1
fi

export YOUR_PATH="/home/jyt/workspace/transfer_models"
export gpu_idx=0
export data_dir="/home/jyt/workspace/fNIRS_models/code_data_tufts/fNIRS2MW/experiment/fNIRS_data/band_pass_filtered/slide_window_data/size_30sec_150ts_stride_03ts/"
export window_size=150
export classification_task="binary"
export scenario="64vs4"  # take care!
export n_epoch=60


buckets=("TestBucket1" "TestBucket2" "TestBucket3" "TestBucket4")
settings64vs4=("64vs4_TestBucket1" "64vs4_TestBucket2" "64vs4_TestBucket3" "64vs4_TestBucket4")
settings16vs4=("16vs4_TestBucket1" "16vs4_TestBucket2" "16vs4_TestBucket3" "16vs4_TestBucket4")
settings4vs4=("4vs4_TestBucket1" "4vs4_TestBucket2" "4vs4_TestBucket3" "4vs4_TestBucket4")

for ((i=0; i<4; i++))
do 
    export bucket=${buckets[i]}
    export setting=${settings64vs4[i]}
    export result_save_rootdir='None'    #"$YOUR_PATH/transfer_BCI/experiments/generic_models/OursT_1_finetune/binary/$scenario/$bucket" 
    export restore_file="$YOUR_PATH/transfer_BCI/experiments/generic_models/OursT_1_pre/binary/$scenario/$bucket" 

    bash $YOUR_PATH/transfer_BCI/generic_models/runs/do_experiment_OursT.slurm
done
