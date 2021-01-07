python expert_trajectories.py   -e ConveyorEnv -s 20210107_0700 -n 500 -c config1
python pretrain.py              -e ConveyorEnv -s 20210107_0700 -c config1
python retrain_callback.py      -e ConveyorEnv -s 20210107_0700 -n 2x2design

python expert_trajectories.py   -e ConveyorEnv -s 20210107_0800 -n 500 -c config2
python pretrain.py              -e ConveyorEnv -s 20210107_0800 -c config2
python retrain_callback.py      -e ConveyorEnv -s 20210107_0800 -n 2x3design

python expert_trajectories.py   -e ConveyorEnv -s 20210107_0900 -n 500 -c config3
python pretrain.py              -e ConveyorEnv -s 20210107_0900 -c config3
python retrain_callback.py      -e ConveyorEnv -s 20210107_0900 -n 3x3design

python expert_trajectories.py   -e ConveyorEnv -s 20210107_1000 -n 500 -c config4
python pretrain.py              -e ConveyorEnv -s 20210107_1000 -c config4
python retrain_callback.py      -e ConveyorEnv -s 20210107_1000 -n 4x3design

python expert_trajectories.py   -e ConveyorEnv -s 20210107_1100 -n 500 -c config5
python pretrain.py              -e ConveyorEnv -s 20210107_1100 -c config5
python retrain_callback.py      -e ConveyorEnv -s 20210107_1100 -n 5x3design