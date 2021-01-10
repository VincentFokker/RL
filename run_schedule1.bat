python expert_trajectories.py   -e ConveyorEnv1 -s 20210110_1000 -n 500 -c config1
python pretrain.py              -e ConveyorEnv1 -s 20210110_1000 -c config1
python retrain_callback.py      -e ConveyorEnv1 -s 20210110_1000 -n 1x2design
python expert_trajectories.py   -e ConveyorEnv1 -s 20210110_1100 -n 500 -c config2
python pretrain.py              -e ConveyorEnv1 -s 20210110_1100 -c config2
python retrain_callback.py      -e ConveyorEnv1 -s 20210110_1100 -n 2x2design
python expert_trajectories.py   -e ConveyorEnv1 -s 20210110_1200 -n 500 -c config3
python pretrain.py              -e ConveyorEnv1 -s 20210110_1200 -c config3
python retrain_callback.py      -e ConveyorEnv1 -s 20210110_1200 -n 2x3design
python expert_trajectories.py   -e ConveyorEnv1 -s 20210110_1300 -n 500 -c config4
python pretrain.py              -e ConveyorEnv1 -s 20210110_1300 -c config4
python retrain_callback.py      -e ConveyorEnv1 -s 20210110_1300 -n 3x3design
python expert_trajectories.py   -e ConveyorEnv1 -s 20210110_1400 -n 500 -c config5
python pretrain.py              -e ConveyorEnv1 -s 20210110_1400 -c config5
python retrain_callback.py      -e ConveyorEnv1 -s 20210110_1400 -n 4x3design
python expert_trajectories.py   -e ConveyorEnv1 -s 20210110_1500 -n 500 -c config6
python pretrain.py              -e ConveyorEnv1 -s 20210110_1500 -c config6
python retrain_callback.py      -e ConveyorEnv1 -s 20210110_1500 -n 5x3design
