# Setup logbook
Overview of the different versions of the Simple_conveyor environment. 
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
<b>Description</b>            : Version without exception handling 

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