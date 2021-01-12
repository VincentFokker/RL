python train_on.py -e ConveyorEnv121 -s 20210113_0000 -n buffer1 -c buffer1
python train_on.py -e ConveyorEnv121 -s 20210113_0030 -n buffer2 -c buffer2
python train_on.py -e ConveyorEnv121 -s 20210113_0100 -n buffer3 -c buffer3
python train_on.py -e ConveyorEnv121 -s 20210113_0130 -n buffer4 -c buffer4
python train_on.py -e ConveyorEnv121 -s 20210113_0200 -n buffer5 -c buffer5
python train_on.py -e ConveyorEnv121 -s 20210113_0230 -n buffer6 -c buffer6
python train_on.py -e ConveyorEnv121 -s 20210113_0300 -n buffer7 -c buffer7
python train_on.py -e ConveyorEnv121 -s 20210113_0330 -n buffer8 -c buffer8
python train_on.py -e ConveyorEnv121 -s 20210113_0400 -n buffer9 -c buffer9
python train_on.py -e ConveyorEnv121 -s 20210113_0430 -n buffer10 -c buffer10
python plotcombiner_b.py -t terminate
python resultmaker_b.py -t buffer -n 100

python train_on.py -e ConveyorEnv121 -s 20210113_0500 -n pipe10 -c pipe10
python train_on.py -e ConveyorEnv121 -s 20210113_0530 -n pipe15 -c pipe15
python train_on.py -e ConveyorEnv121 -s 20210113_0600 -n pipe20 -c pipe20
python train_on.py -e ConveyorEnv121 -s 20210113_0630 -n pipe25 -c pipe25
python train_on.py -e ConveyorEnv121 -s 20210113_0700 -n pipe30 -c pipe30
python train_on.py -e ConveyorEnv121 -s 20210113_0730 -n pipe35 -c pipe35
python train_on.py -e ConveyorEnv121 -s 20210113_0800 -n pipe40 -c pipe40
python train_on.py -e ConveyorEnv121 -s 20210113_0830 -n pipe45 -c pipe45
python train_on.py -e ConveyorEnv121 -s 20210113_0900 -n pipe50 -c pipe50
python plotcombiner_p.py -t terminate
python resultmaker_p.py -t pipe -n 100