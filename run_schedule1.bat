python retrain_callback.py -e ConveyorEnv11 -s 20210108_1400 -n LongerRun
python retrain_callback.py -e ConveyorEnv11 -s 20210108_1500 -n LongerRun
python retrain_callback.py -e ConveyorEnv11 -s 20210108_1600 -n LongerRun

python expert_trajectories.py   -e ConveyorEnv11 -s 20210109_1800 -n 500 -c pipe40
python pretrain.py              -e ConveyorEnv11 -s 20210109_1800 -c pipe40
python retrain_callback.py      -e ConveyorEnv11 -s 20210109_1800 -n pipe40

python expert_trajectories.py   -e ConveyorEnv11 -s 20210109_1900 -n 500 -c pipe45
python pretrain.py              -e ConveyorEnv11 -s 20210109_1900 -c pipe45
python retrain_callback.py      -e ConveyorEnv11 -s 20210109_1900 -n pipe45

python expert_trajectories.py   -e ConveyorEnv11 -s 20210109_2000 -n 500 -c pipe50
python pretrain.py              -e ConveyorEnv11 -s 20210109_2000 -c pipe50
python retrain_callback.py      -e ConveyorEnv11 -s 20210109_2000 -n pipe50

python expert_trajectories.py   -e ConveyorEnv1 -s 20210109_1300 -n 500 -c config1
python pretrain.py              -e ConveyorEnv1 -s 20210109_1300 -c config1
python retrain_callback.py      -e ConveyorEnv1 -s 20210109_1300 -n 1x2design
python expert_trajectories.py   -e ConveyorEnv1 -s 20210109_1300 -n 500 -c config2
python pretrain.py              -e ConveyorEnv1 -s 20210109_1300 -c config2
python retrain_callback.py      -e ConveyorEnv1 -s 20210109_1300 -n 2x2design
python expert_trajectories.py   -e ConveyorEnv1 -s 20210109_1300 -n 500 -c config3
python pretrain.py              -e ConveyorEnv1 -s 20210109_1300 -c config3
python retrain_callback.py      -e ConveyorEnv1 -s 20210109_1300 -n 2x3design
python expert_trajectories.py   -e ConveyorEnv1 -s 20210109_1300 -n 500 -c config4
python pretrain.py              -e ConveyorEnv1 -s 20210109_1300 -c config4
python retrain_callback.py      -e ConveyorEnv1 -s 20210109_1300 -n 3x3design
python expert_trajectories.py   -e ConveyorEnv1 -s 20210109_1300 -n 500 -c config5
python pretrain.py              -e ConveyorEnv1 -s 20210109_1300 -c config5
python retrain_callback.py      -e ConveyorEnv1 -s 20210109_1300 -n 4x3design
python expert_trajectories.py   -e ConveyorEnv1 -s 20210109_1300 -n 500 -c config6
python pretrain.py              -e ConveyorEnv1 -s 20210109_1300 -c config6
python retrain_callback.py      -e ConveyorEnv1 -s 20210109_1300 -n 5x3design

python train_callback.py -e ConveyorEnv1 -s 20210109_1900 -n 1x2design_wp -c config1
python train_callback.py -e ConveyorEnv1 -s 20210109_2000 -n 2x2design_wp -c config2
python train_callback.py -e ConveyorEnv1 -s 20210109_2100 -n 2x3design_wp -c config3
python train_callback.py -e ConveyorEnv1 -s 20210109_2200 -n 3x3design_wp -c config4
python train_callback.py -e ConveyorEnv1 -s 20210109_2300 -n 4x3design_wp -c config5
python train_callback.py -e ConveyorEnv1 -s 20210109_2400 -n 5x3design_wp -c config6

python train_callback.py -e ConveyorEnv11 -s 20210110_0000 -n buffer1_wp -c buffer1
python train_callback.py -e ConveyorEnv11 -s 20210110_0100 -n buffer2_wp -c buffer2
python train_callback.py -e ConveyorEnv11 -s 20210110_0200 -n buffer3_wp -c buffer3
python train_callback.py -e ConveyorEnv11 -s 20210110_0300 -n buffer4_wp -c buffer4
python train_callback.py -e ConveyorEnv11 -s 20210110_0400 -n buffer5_wp -c buffer5
python train_callback.py -e ConveyorEnv11 -s 20210110_0500 -n buffer6_wp -c buffer6
python train_callback.py -e ConveyorEnv11 -s 20210110_0600 -n buffer7_wp -c buffer7
python train_callback.py -e ConveyorEnv11 -s 20210110_0700 -n buffer8_wp -c buffer8
python train_callback.py -e ConveyorEnv11 -s 20210110_0800 -n buffer9_wp -c buffer9
python train_callback.py -e ConveyorEnv11 -s 20210110_0900 -n buffer10_wp -c buffer10
