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


for SubjectId_of_interest in 45 24 36 71
do
    export experiment_dir="YOUR_PATH/fNIRS-mental-workload-classifiers/experiments/generic_models/RandomForest/binary/4vs4/TestBucket8/$SubjectId_of_interest"
    
    echo "Current experiment_dir is $experiment_dir"
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < YOUR_PATH/fNIRS-mental_workload-classifiers/synthesizing_results/generic_models/binary/RandomForest/synthesize_hypersearch_RF_for_a_subject.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash YOUR_PATH/fNIRS-mental_workload-classifiers/synthesizing_results/generic_models/binary/RandomForest/synthesize_hypersearch_RF_for_a_subject.slurm
    fi
    
done