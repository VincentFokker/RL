python expert_trajectories.py   -e ConveyorEnv -s 20210109_0400 -n 500 -c config5
python pretrain.py              -e ConveyorEnv -s 20210109_0400 -c config6
python retrain_callback.py      -e ConveyorEnv -s 20210109_0400 -n 3x3design_33reward

python expert_trajectories.py   -e ConveyorEnv -s 20210109_0500 -n 500 -c config6
python pretrain.py              -e ConveyorEnv -s 20210109_0500 -c config7
python retrain_callback.py      -e ConveyorEnv -s 20210109_0500 -n 4x3design_33reward

python expert_trajectories.py   -e ConveyorEnv -s 20210109_0600 -n 500 -c config7
python pretrain.py              -e ConveyorEnv -s 20210109_0600 -c config8
python retrain_callback.py      -e ConveyorEnv -s 20210109_0600 -n 4x3design_11reward

python expert_trajectories.py   -e ConveyorEnv -s 20210109_0700 -n 500 -c config8
python pretrain.py              -e ConveyorEnv -s 20210109_0700 -c config9
python retrain_callback.py      -e ConveyorEnv -s 20210109_0700 -n 5x3design_11reward

python expert_trajectories.py   -e ConveyorEnv -s 20210109_0800 -n 500 -c config9
python pretrain.py              -e ConveyorEnv -s 20210109_0800 -c config10
python retrain_callback.py      -e ConveyorEnv -s 20210109_0800 -n 5x3design_33reward