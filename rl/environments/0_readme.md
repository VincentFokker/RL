# Setup logbook
Overview of the different versions of the Simple_conveyor environment. 
General syntax for training on an environment: \
<code>python train.py -e <ENVIRONMENT_NAME> -s <SUB_DIR> -n <NAME_GIVEN> </code> \
<code>python train.py -e simple_conveyor -s PPO2 -n 1M5Worker</code> 

TEST a trained model: \
<code> python test.py -e <ENVIRONMENT_NAME> -s <SUB_DIR> -n 0 --render</code> \
<code> python test.py -e simple_conveyor -s PPO2 -n 0 --render</code>
## Changelog 
Newest to oldest. Based on the Environment name.

### V.7 - Simple_conveyor_7
<b>Description</b>            : Combination of V5 and V6 \
<b>New features</b> :  
- Decreased the observation space to a size [3*amount_of_gtp, ], only observing the next up 3 items demanded at a queue (gtp queue demand)
-  Warm start is the default now
- instead of queueing up the items at output lanes, each step this action is evaluated, if not possible, a punishment is given.
- it is no longer possible to output more items of a certain type then there are in the init_queue (gtp queue demand).


<b>File(names)</b>         : 
- environment: <pre> /environments/Simple_conveyor_7.py  </pre>
- config: <pre> /config/Simple_conveyor_7.yml </pre>
\
\
\
\.

### V.6 - Simple_conveyor_6
<b>Description</b>            : Based on V4, much smaller observation space \
<b>New features</b> :  
- Decreased the observation space to a size [3*amount_of_gtp, ], only observing the next up 3 items demanded at a queue (gtp queue demand)


<b>File(names)</b>         : 
- environment: <pre> /environments/Simple_conveyor_6.py  </pre>
- config: <pre> /config/Simple_conveyor_6.yml </pre>
\
\
\
\.
### V.5 - Simple_conveyor_5
<b>Description</b>            : Based on V4, different introduction policy \
<b>New features</b> :  
- Warm start is the default now
- instead of queueing up the items at output lanes, each step this action is evaluated, if not possible, a punishment is given.
- it is no longer possible to output more items of a certain type then there are in the init_queue (gtp queue demand).


<b>File(names)</b>         : 
- environment: <pre> /environments/Simple_conveyor_5.py  </pre>
- config: <pre> /config/Simple_conveyor_5.yml </pre>
\
\
\
\.
### V.4 - Simple_conveyor_4
<b>Description</b>            : Based on V3, different introduction policy \
<b>New features</b> :  
- Warm start is the default now
- instead of queueing up the items at output lanes, each step this action is evaluated, if not possible, a punishment is given.


<b>File(names)</b>         : 
- environment: <pre> /environments/Simple_conveyor_4.py  </pre>
- config: <pre> /config/Simple_conveyor_4.yml </pre>
\
\
\
\.
### V.3 - Simple_conveyor_3
<b>Description</b>            : Based on version V2, includes a warm start \
<b>New features</b> :  
- Environment includes a warm start, preventing punishment for empty queue from the beginning.


<b>File(names)</b>         : 
- environment: <pre> /environments/Simple_conveyor_3.py  </pre>
- config: <pre> /config/Simple_conveyor_3.yml </pre>
\
\
\
\.
### V.2 - Simple_conveyor_2_intobs
<b>Description</b>            : Similar version to V2, different observation space \
<b>New features</b> :  
- Observation space is now represented in integers instead of binary code.

<b>File(names)</b>         : 
- environment: <pre> /environments/Simple_conveyor_2_intobs.py  </pre>
- config: <pre> /config/Simple_conveyor_2_intobs.yml </pre>
\
\
\
\.
### V.2 - Simple_conveyor_2
<b>Description</b>            : Version without exception handling . \
<b>New features</b> :  
- Added automatic scaling of the observation space based on the amount of GtP used. 
- included the full observation of the conveyor
- changed negative reward to only obtain negative reward for cycling items.

<b>File(names)</b>         : 
- environment: <pre> /environments/Simple_conveyor_2.py  </pre>
- config: <pre> /config/Simple_conveyor_2.yml </pre>

<b>Best model </b> \
<code>
python test.py -e simple_conveyor_2 -s 20200910_1105 -n 0 --render \
python retrain.py -e simple_conveyor_2 -s 20200910_1105 -n 0 -t</code> 
- shows behavior off more preference for step 0
- now leaves queues empty, should add punishment for this.



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
    
