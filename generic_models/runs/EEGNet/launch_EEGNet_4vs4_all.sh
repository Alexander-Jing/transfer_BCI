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

export YOUR_PATH="/home/jyt/workspace/fNIRS_models/code_data_tufts"
export gpu_idx=0
export data_dir="$YOUR_PATH/fNIRS2MW/experiment/fNIRS_data/band_pass_filtered/slide_window_data/size_30sec_150ts_stride_03ts/"
export window_size=150
export classification_task="binary"
export scenario="4vs4"
export n_epoch=600
export restore_file="None"

testbuckets=("TestBucket1" "TestBucket2" "TestBucket3" "TestBucket4" "TestBucket5" "TestBucket6" "TestBucket7" "TestBucket8" "TestBucket9" "TestBucket10" "TestBucket11" "TestBucket12" "TestBucket13" "TestBucket14" "TestBucket15" "TestBucket16" "TestBucket17")
settings64vs4=("64vs4_TestBucket1" "64vs4_TestBucket2" "64vs4_TestBucket3" "64vs4_TestBucket4" "64vs4_TestBucket5" "64vs4_TestBucket6" "64vs4_TestBucket7" "64vs4_TestBucket8" "64vs4_TestBucket9" "64vs4_TestBucket10" "64vs4_TestBucket11" "64vs4_TestBucket12" "64vs4_TestBucket13" "64vs4_TestBucket14" "64vs4_TestBucket15" "64vs4_TestBucket16" "64vs4_TestBucket17")
settings4vs4=("4vs4_TestBucket1" "4vs4_TestBucket2" "4vs4_TestBucket3" "4vs4_TestBucket4" "4vs4_TestBucket5" "4vs4_TestBucket6" "4vs4_TestBucket7" "4vs4_TestBucket8" "4vs4_TestBucket9" "4vs4_TestBucket10" "4vs4_TestBucket11" "4vs4_TestBucket12" "4vs4_TestBucket13" "4vs4_TestBucket14" "4vs4_TestBucket15" "4vs4_TestBucket16" "4vs4_TestBucket17")

for ((i=0; i<17; i++))
do
    export bucket=${testbuckets[i]}
    export setting=${settings4vs4[i]}
    export result_save_rootdir="$YOUR_PATH/fNIRS-mental-workload-classifiers/experiments/generic_models/EEGNet/binary/$scenario/$bucket" 
    bash $YOUR_PATH/fNIRS-mental-workload-classifiers/generic_models/runs/do_experiment_EEGNet.slurm
done

