# Setup logbook
Overview of the different versions of the Simple_conveyor environment. 
General syntax for training on an environment: \
<code>python train.py -e <ENVIRONMENT_NAME> -s <SUB_DIR> -n <NAME_GIVEN> </code> \
<code>python train.py -e simple_conveyor -s PPO2 -n 1M5Worker</code> 

TEST a trained model: \
<code> python test.py -e <ENVIRONMENT_NAME> -s <SUB_DIR> -n 0 --render</code> \
<code> python test.py -e simple_conveyor -s PPO2 -n 0 --render</code>
## Version overview

### V.1
<b>Description</b>            : Version without exception handling 

<b>File(names)</b>         : 
- environment: <pre> /environments/Simple_conveyor.py  </pre>
- config: <pre> /config/Simple_conveyor.yml </pre>



<b>State observation</b>   :
- queue demand &nbsp; &nbsp; &nbsp; &nbsp; - &nbsp; <code>Init_queue </code>
- In queue &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; - &nbsp; <code>In_queue </code>
- 1/2 of the conveyor &nbsp; - &nbsp; <code>type_map_obs </code>
    -   right lane, bottom lane
        - information about type of carrier, time in the system

<b>Rewards </b> :
- Positive reward for proper delivery of order carrier at GtP, as soon as order carrier is processed.
    - <code>travelpath_to_gtp_reward</code>
- negative reward for each step for each item on the conveyor.
    - <code>negative_reward_per_step</code>
    
### V.2
<b>Description</b>            : Version without exception handling . \
<b>New features</b> :  
- Added automatic scaling of the observation space based on the amount of GtP used. 
- included the full observation of the conveyor
- changed negative reward to only obtain negative reward for cycling items.

<b>File(names)</b>         : 
- environment: <pre> /environments/Simple_conveyor_1.py  </pre>
- config: <pre> /config/Simple_conveyor_1.yml </pre>



<b>State observation</b>   :
- queue demand &nbsp; &nbsp; &nbsp; &nbsp; - &nbsp; <code>Init_queue </code>
- In queue &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; - &nbsp; <code>In_queue </code>
- 1/2 of the conveyor &nbsp; - &nbsp; <code>type_map_obs </code>
    -   right lane, <b>left lane</b>, bottom lane, <b>top lane</b>
        - information about type of carrier, time in the system

<b>Rewards </b> :
- Positive reward for proper delivery of order carrier at GtP, as soon as order carrier is processed.
    - <code>travelpath_to_gtp_reward</code>
- negative reward for each step for each item on the conveyor.
    - <code>negative_reward_per_step</code>