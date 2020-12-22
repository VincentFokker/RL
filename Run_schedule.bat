python train.py -e AbstractConveyor4 -s 20201221_2230 -n RewardPerEpisode

python train5.py -e AbstractConveyor5 -s 20201221_2300 -n RewardPerStep -t
python plotmaker.py -e AbstractConveyor5 -s 20201221_2300
python train.py -e AbstractConveyor5 -s 20201221_2330 -n RewardPerStep

PAUSE