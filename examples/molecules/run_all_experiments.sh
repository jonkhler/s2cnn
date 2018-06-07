#!/bin/zsh
runid=0;
for run in 0 1 2 3 4;
do
    for strat in 0 1 2 3 4;
    do
        echo "starting run $run for strat $strat"
        python3 run_experiment.py --test_strat ${strat} > logs/default_settings_strat_${strat}_run_${run}.txt
    done;
done;

