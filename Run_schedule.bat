python train2.py -e AbstractConveyor1 -s 20201128_1200 -n 7_0_5_13_11_8 	 -c config1
python plotmaker.py -e AbstractConveyor1 -s 20201128_1200
python train2.py -e AbstractConveyor1 -s 20201128_1250 -n 15_6_6_15_9_6 	 -c config2
python plotmaker.py -e AbstractConveyor1 -s 20201128_1250
python train2.py -e AbstractConveyor1 -s 20201128_1340 -n 7_0_5_13_11_8 	 -c config3
python plotmaker.py -e AbstractConveyor1 -s 20201128_1340

PAUSE