python expert_trajectories.py -e ConveyorEnv12 -s 20210113_0000 -c term5 -n 500
python pretrain.py -e ConveyorEnv12 -s 20210113_0000 -c term5
python train_on.py -e ConveyorEnv12 -s 20210113_0000 -n 2x3design -t -c term5

python expert_trajectories.py -e ConveyorEnv12 -s 20210113_0100 -c term6 -n 500
python pretrain.py -e ConveyorEnv12 -s 20210113_0100 -c term6
python train_on.py -e ConveyorEnv12 -s 20210113_0100 -n 3x3design -t -c term6

python expert_trajectories.py -e ConveyorEnv12 -s 20210113_0200 -c term7 -n 500
python pretrain.py -e ConveyorEnv12 -s 20210113_0200 -c term7
python train_on.py -e ConveyorEnv12 -s 20210113_0200 -n 2x3design -t -c term7

python expert_trajectories.py -e ConveyorEnv12 -s 20210113_0300 -c term8 -n 500
python pretrain.py -e ConveyorEnv12 -s 20210113_0300 -c term8
python train_on.py -e ConveyorEnv12 -s 20210113_0300 -n 3x3design -t -c term8

python expert_trajectories.py -e ConveyorEnv12 -s 20210113_0400 -c term9 -n 500
python pretrain.py -e ConveyorEnv12 -s 20210113_0400 -c term9
python train_on.py -e ConveyorEnv12 -s 20210113_0400 -n 4x3design -t -c term9

python expert_trajectories.py -e ConveyorEnv12 -s 20210113_0500 -c term10 -n 500
python pretrain.py -e ConveyorEnv12 -s 20210113_0500 -c term10
python train_on.py -e ConveyorEnv12 -s 20210113_0500 -n 5x3design -t -c term10