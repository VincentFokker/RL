########################################################################################
# New version with right reward reset build in.
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
from copy import copy, deepcopy
import seaborn as sns
import random
import logging
import gym

# TODO: resize the observation space
# TODO: larger warm start
# TODO: Revise the observation

#CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements
logging.basicConfig(level=logging.INFO)

class MultiConveyor1(gym.Env):

######## INITIALIZATION OF VARIABLES ###############################################################################################################
    def __init__(self, config, **kwargs):
        """initialize states of the variables, the lists used"""

        # init config
        self.config = config['environment']

        self.amount_of_gtps = self.config['amount_gtp']
        self.amount_of_outputs = self.config['amount_output']
        self.gtp_buffer_size = self.config['gtp_buffer_size']
        self.exception_occurence = self.config['exception_occurence']  # % of the times, an exception occurs
        self.process_time_at_GTP = self.config['process_time_at_GTP']  # takes 30 timesteps
        self.max_time_in_system = self.config['max_time_in_system']
        self.window_name = 'Conveyor render'
        self.percentage_small_carriers = self.config['percentage_small_carriers']
        self.percentage_medium_carriers = self.config['percentage_medium_carriers']
        self.percentage_large_carriers = self.config['percentage_large_carriers']
        self.in_que_observed = self.config['in_que_observed']
        self.termination_condition = self.config['termination_condition']
        self.observation_shape = self.config['observation_shape']
        self.max_cycle_count = self.config['max_cycle_count']
        self.amount_to_process = self.config['amount_to_process']



        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(32)

        #determination of observation_space:
        self.shape = 0
        if 1 in self.observation_shape:
            self.shape += 2*((self.amount_of_gtps*4) + 6 + 2*self.amount_of_outputs)
        if 2 in self.observation_shape:
            self.shape += self.amount_of_outputs
        if 3 in self.observation_shape:
            self.shape += self.amount_of_gtps
        if 4 in self.observation_shape:
            self.shape += 2 * self.in_que_observed * self.amount_of_gtps
        if 5 in self.observation_shape:
            self.shape += 1
        if 6 in self.observation_shape:
            self.shape += 1
        if 7 in self.observation_shape:
            self.shape += 1
        if 8 in self.observation_shape:
            self.shape += self.amount_of_gtps
        if 9 in self.observation_shape:
            self.shape += self.amount_of_gtps
        if 10 in self.observation_shape:
            self.shape += self.amount_of_gtps
        if 11 in self.observation_shape:
            self.shape += 2*self.amount_of_outputs
        if 12 in self.observation_shape:
            self.shape += self.amount_of_gtps * self.amount_of_outputs
        if 13 in self.observation_shape:
            self.shape += 2* 2 * ((self.amount_of_gtps * 4) + 6 + 2 * self.amount_of_outputs) + 2*8
        if 14 in self.observation_shape:
            self.shape += 2 * self.amount_of_gtps
        logging.info('Action Space: {}, Observation shape: {}'.format(self.amount_of_outputs * self.amount_of_gtps + 1, self.shape))

        #self.shape = 2*((self.amount_of_gtps*4) + 13) + (self.in_que_observed * self.amount_of_gtps * 2) + (2 * self.amount_of_gtps)
        self.observation_space = gym.spaces.Box(shape=(self.shape, ),
                                                high=1, low=0,
                                                dtype=np.uint8)

        #init queues
        self.queues = [random.choices(np.arange(1,self.amount_of_outputs+1), [0.33, 0.34, 0.33], k=self.gtp_buffer_size) for item in range(self.amount_of_gtps)] # generate random queues
        logging.debug("queues that are initialized: {}".format(self.queues))
        self.init_queues = copy(self.queues)
        self.demand_queues = copy(self.queues)
        self.in_queue = [[] for item in range(len(self.queues))]

        ## rewards
        # All punishments found by: tag:punishment
        self.negative_reward_for_cycle = self.config['negative_reward_for_cycle']
        self.negative_reward_for_flooding = self.config['negative_reward_for_flooding']
        self.negative_reward_full_queue = self.config['negative_reward_full_queue']
        self.negative_reward_for_empty_queue = self.config['negative_reward_for_empty_queue']
        self.positive_reward_for_divert = self.config['positive_reward_for_divert']
        self.negative_reward_for_invalid = self.config['negative_reward_for_invalid']
        self.punishment_for_not_taken = self.config['punishment_for_not_taken']
        self.negative_reward_for_wrong_demand = self.config['negative_reward_for_wrong_demand']

        #tracers
        self.amount_of_items_on_conv = 0
        self.amount_of_items_in_sys = 0
        self.remaining_demand = 0
        self.amount_of_orders_processed = 0
        self.positive_reward = 0
        self.negative_reward = 0
        self.cycle_count = 0
        self.run_count = 0
        self.same_action_count = 0
        self.previous_action = 0
        self.items_processed = 0
        self.termination_cases = {}
        self.termination_cases['demand_satisfied'] = 0
        self.termination_cases['queues_satisfied'] = 0
        self.termination_cases['amounts_processed'] = 0
        self.termination_cases['too_much_steps'] = 0
        self.termination_cases['too_much_sim_steps'] = 0
        self.termination_cases['cycle_count'] = 0

        #gym related part
        self.reward = 0
        self.step_reward_p = 0
        self.step_reward_n = 0
        self.total_travel = 0
        self.terminate = False
        self.episode= 0
        self.steps = 0
        self.O_state = 0

        self.create_window()

        #colors
        self.pallette = (np.asarray(sns.color_palette("Reds", self.amount_of_outputs)) * 255).astype(int)

        # build env
        self.empty_env = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)
        self.image = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)

        #make actionsmap
        action = 0
        self.actions_map = {}
        for j in range(0, 2):
            for k in range(0, 2):
                for l in range(0, 2):
                    for i in range(0, 4):
                        self.actions_map[action] = [j, k, l, i]
                        action += 1

        #define where the operators, diverts and outputs of the GTP stations are
        self.operator_locations = [[i, self.empty_env.shape[0]-1] for i in range(4,self.amount_of_gtps*4+1,4)][::-1]
        self.output_locations = [[i,7]for i in range(self.empty_env.shape[1]-self.amount_of_outputs*2-1,self.empty_env.shape[1]-2,2)]
        self.diverter_locations = [[i, 7] for i in range(4,self.amount_of_gtps*4+1,4)][::-1]
        self.merge_locations = [[i-1, 7] for i in range(4,self.amount_of_gtps*4+1,4)][::-1]
        logging.debug("operator locations: {}".format(self.operator_locations))
        logging.debug("output locations: {}".format( self.output_locations))
        logging.debug("diverter locations: {}".format( self.diverter_locations))
        logging.debug("Merge locations: {}".format(self.merge_locations))

        #initialize divert points: False=no diversion, True=diversion
        self.D_states = {}
        for i in range(1,len(self.diverter_locations)+1):
            self.D_states[i] = False
        
        #D conditions
        self.D_condition_1 = {}
        for i in range(1,len(self.diverter_locations)+1):
            self.D_condition_1[i] = False

        self.D_condition_2 = {}
        for i in range(1,len(self.diverter_locations)+1):
            self.D_condition_2[i] = False
    

        #initialize output points
        self.O_states = {}
        for i in range(1,len(self.output_locations)+1):
            self.O_states[i] = 0

        # initialize transition points: 0=no transition, 1=transition
        self.T_states = {}
        for i in range(1,len(self.operator_locations)+1):
            self.T_states[i] = False

        #initialize merge points
        self.M_states = {}
        for i in range(1,len(self.merge_locations)+1):
            self.M_states[i] = False

####### FOR SIMULATION ONLY 
        self.W_times = {}
        for i in range(1,len(self.operator_locations)+1):
            self.W_times[i] = self.process_time_at_GTP +8*self.amount_of_gtps -5
        logging.debug("Process times at operator are:{}".format(self.W_times))
####### FOR SIMULATION ONLY
        self.condition_to_transfer = False
        self.condition_to_process = False

        #initialize conveyor memory
        self.items_on_conv = []        
        self.carrier_type_map = np.zeros((self.empty_env.shape[0],self.empty_env.shape[1],1))

        ######## Do a warm_start
        self.do_warm_start(3)

    def do_warm_start(self, x):
        for _ in range(x):
            self.warm_start()

    def warm_start(self):
        # add items to queues
        for _ in self.operator_locations:
            self.in_queue[self.operator_locations.index(_)].append(self.init_queues[self.operator_locations.index(_)][0])

            # add to items_on_conv
            #self.items_on_conv.append([_, self.init_queues[self.operator_locations.index(_)][0], 45/self.max_time_in_system])
            if len(self.in_queue[0]) == 1:
                self.items_on_conv.append(
                    [_, self.init_queues[self.operator_locations.index(_)][0], 0])
            elif len(self.in_queue[0]) == 2:
                self.items_on_conv.append(
                    [[_[0], _[1] - 1], self.init_queues[self.operator_locations.index(_)][0],0])
            elif len(self.in_queue[0]) == 3:
                self.items_on_conv.append(
                    [[_[0], _[1] - 2], self.init_queues[self.operator_locations.index(_)][0], 0])
            elif len(self.in_queue[0]) == 4:
                self.items_on_conv.append(
                    [[_[0], _[1] - 3], self.init_queues[self.operator_locations.index(_)][0], 0])

        # remove from init
        self.init_queues = [item[1:] for item in self.init_queues]

#### Generate the visual conveyor ##########################################################################################################

    def generate_env(self, no_of_gtp, no_of_output):
        """returns empty env, with some variables about the size, lanes etc."""
        empty = np.zeros((15, 4*no_of_gtp + 4 + 3*no_of_output +2, 3))                   # height = 15, width dependent on amount of stations
        for i in range(2,empty.shape[1]-2):
            empty[2][i]=(255,255,255)                               #toplane = 2
            empty[7][i]=(255,255,255)                               #bottom lane = 7
        for i in range(2,8):
            empty[i][1]=(255,255,255)                               #left lane
            empty[i][empty.shape[1]-2]=(255,255,255)                #right lane

        for i in range(8,empty.shape[0]): 
            for j in range(4,no_of_gtp*4+1,4):                      #GTP lanes
                empty[i][j] = (255,255,255)                         #Gtp in
                empty[i][j-1] = (250, 250,250)                      #gtp out
            for j in range(empty.shape[1]-no_of_output*2-1,empty.shape[1]-2,2):   #order carrier lanes
                empty[i][j] = (255,255,255)
        for i in range(4,no_of_gtp*4+1, 4):                         #divert and merge points
            empty[7][i-1] = (255, 242, 229)                         #merge
            empty[7][i] = (255, 242, 229)                           #divert

        for i in range(empty.shape[1]-no_of_output*2-1,empty.shape[1]-2,2):       #output points
            empty[7][i] = (255, 242, 229)

        return empty

###### HELPER FUNCTIONS ###########################################################################################################################
        
    def update_queues(self, quenr, variable):
        'For a given queue 1-3, add a variable (1,2,3)'
        for i in range(self.amount_of_gtps):
            if quenr == i+1:
                self.init_queues[i].append(variable)
            
    def remove_from_queue(self, quenr):
        'For a given queue 1-3, remove the first in the queue'
        for i in range(self.amount_of_gtps):
            if quenr == i+1:
                self.init_queues[i]= self.init_queues[i][1:]

    def add_to_in_que(self, que_nr, to_add):
        'for a given queue, add item to the queue'
        self.in_queue[que_nr].append(to_add)
        
    def encode(self, var):
        """encodes categorical variables 0-3 to binary"""
        return (0,0) if var == 0 else (0,1) if var == 1 else (1,0) if var == 2 else (1,1) if var ==3 else var

#####################################################################################
## Make Observation
#
    def make_q_obs(self, cut):
        '''Observation for agent at GTP 1'''

        #amount of items in each queue
        in_queue = [len(item) * 1 / 7 for item in self.in_queue[cut:]]
        in_queue = np.array(in_queue).flatten()

        #Demand at the queue
        init = []
        for item in self.init_queues[cut:]:
            init1 = item[:self.in_que_observed]
            init.append(init1 + [0] * (self.in_que_observed - len(init1)))
        init = list(np.array(init).flatten())
        # binary encoding of the categorical variables
        init = np.array([self.encode(item) for item in init]).flatten()

        obs = np.append(in_queue, init)

        return obs

    def make_observation(self):
        '''Builds the observation from the available variables'''

        ### 1 . For the obeservation of the conveyor ########################################################################
        self.carrier_type_map_obs = np.zeros((self.empty_env.shape[0], self.empty_env.shape[1], 1)).astype(float)
        for item in self.items_on_conv:
            self.carrier_type_map_obs[item[0][1]][item[0][0]] = item[1]

        type_map_obs = self.carrier_type_map_obs[2:8, 1:-1]                    # cut padding #for the carrier type
        carrier_type_map_obs = type_map_obs[-1]        #Only observe bottom lane #top and bottom lane for the carrier type
        type_map_obs = np.array([self.encode(item) for item in list(carrier_type_map_obs)]).flatten()   #binary encoded memory for the type
        logging.debug(carrier_type_map_obs)
        # TODO: return:type_map_obs

        ###  2. Occupation of the The output points ########################################################################
        output_points = carrier_type_map_obs[-2*self.amount_of_outputs:][::2]  ## returns: array([[3.],[3.],[3.]])                         # 3
        output_points = np.array([1 if item != 0 else 0 for item in output_points]) #Returns array(1, 1, 1)
        logging.debug(output_points)
        # TODO: return: output_points

        ### 14. Occupation type of the divert points #####################################################################
        divert_points = carrier_type_map_obs[3:self.amount_of_gtps * 4][::4]  ## returns: array([[3.],[3.],[3.]])                         # 3
        divert_points = np.array([self.encode(item) for item in divert_points]).flatten()
        # TODO: return: divert_points

        ### 3. For the observation of the items in queue ##################################################################
                    #length of each queue (how full)            #some indicator of how long it takes to process this full queue (consider 1- x)
        in_queue = [len(item)* 1/7 for item in self.in_queue]
        in_queue = np.array(in_queue).flatten()

        # TODO: return: in_queue
        ### 4. For the observation of the demand of the GtP Queue #########################################################
        #make the init list
        init = []
        for item in self.init_queues:
            init1 = item[:self.in_que_observed]
            init.append(init1 + [0] * (self.in_que_observed - len(init1)))
        init = list(np.array(init).flatten())
        # binary encoding of the categorical variables
        init = np.array([self.encode(item) for item in init]).flatten()
        logging.debug('init lenght = {}'.format(len(init)))

        # TODO: return: init
        ### 5 . Amount of items on the conveyor ############################################################################
        amount_on_conv = len([item[1] for item in self.items_on_conv if item[0][1] < 8])
        treshhold = 3 * self.amount_of_gtps
        if amount_on_conv > treshhold:
            var = 1
        elif amount_on_conv <= treshhold:
            var = amount_on_conv * 1 / treshhold

        # TODO: return: var
        ####  6. Cycle count ###############################################################################################
        cycle_factor = self.cycle_count / self.max_cycle_count
        # TODO: return: cycle_factor

        ### 7. usability var ############################################################################################
        tot_in_queue = 0
        tot_on_conv = 0
        usability_var = 0
        for queue in self.init_queues:
            for i in range(self.amount_of_outputs):
                amount_in_queue = len([item for item in self.init_queues[0] if item == i + 1])
                tot_in_queue += amount_in_queue
                on_conv = len([item[1] for item in self.items_on_conv if
                               item[0][1] < 8 and item[1] == i + 1 and item[2] == self.init_queues.index(queue) + 1])
                tot_on_conv += on_conv
                if amount_in_queue - on_conv >= 0:
                    indic = 1
                    usability_var += indic
                elif amount_in_queue - on_conv < 0:
                    indic = amount_in_queue / on_conv
                    usability_var += indic
        usability = usability_var / self.amount_of_outputs
        # TODO: return: usability

        ### 8. remaining processingtime queue #########################################################################
        remaining_processtime = [sum(item)* 1/(self.amount_of_outputs*7) for item in self.in_queue]
        remaining_processtime = np.array(remaining_processtime).flatten()

        #TODO: return: remaining_processtime

        ##### 9. Var if queues can still take items ########################################################
        cantake = []
        isempty = []
        for queue in self.in_queue:
            if len(queue) < 7:
                cantake.append(1)
            elif len(queue) == 7:
                cantake.append(0)
        #TODO: return: cantake

        ##### 10. Var if queue is lower then balance ##################################################################
            balance = 4
            if len(queue) < balance:
                isempty.append(1)
            elif len(queue) >= balance:
                isempty.append(0)
        # TODO: return: isempty

        #### 11. amount of items in lead #########################################################################
        bottom_conv = [item[1] for item in self.items_on_conv if item[0][1] < 8]
        info = []
        for i in range(1, self.amount_of_outputs + 1):
            if i in bottom_conv:
                info.append(1)
                info.append(len([item for item in bottom_conv if item == i]) / (
                            2 * ((self.amount_of_gtps * 4) + 6 + 7 + 2 * self.amount_of_outputs)))
            else:
                info.append(0)
                info.append(0)
        info = np.array(info)

        # TODO: return: info
        ### 13. For the FULL obeservation of the conveyor ##############################################################
        self.carrier_type_map_obs = np.zeros((self.empty_env.shape[0], self.empty_env.shape[1], 1)).astype(float)

        for item in self.items_on_conv:
            self.carrier_type_map_obs[item[0][1]][item[0][0]] = item[1]

        # cut padding
        type_map_obs1 = self.carrier_type_map_obs[2:8, 1:-1]  # for the carrier type

        # row 0 and -1 (top and bottom)
        conv_top_bottom1 = np.append(type_map_obs1[0], type_map_obs1[-1])  # top and bottom lane for the carrier type

        # top and bottom lane for the time in the system
        # left and right lane
        conv_left_right1 = np.append(type_map_obs1[1:-1][:, 0],
                                     type_map_obs1[1:-1][:, -1])  # left and right lane for carrier type

        # together
        carrier_type_map_obs1 = np.append(conv_top_bottom1, conv_left_right1)  # full circle for the carrier type

        full_conveyor = np.array(
            [self.encode(item) for item in list(carrier_type_map_obs1)]).flatten()  # binary encoded memory for the type

        ### Combine All to one array ###################################################################################

        obs = np.array([])
        if 1 in self.observation_shape:
            obs = np.append(obs, type_map_obs)
        if 2 in self.observation_shape:
            obs = np.append(obs, output_points)
        if 3 in self.observation_shape:
            obs = np.append(obs, in_queue)
        if 4 in self.observation_shape:
            obs = np.append(obs, init)
        if 5 in self.observation_shape:
            obs = np.append(obs, var)
        if 6 in self.observation_shape:
            obs = np.append(obs, cycle_factor)
        if 7 in self.observation_shape:
            obs = np.append(obs, usability)
        if 8 in self.observation_shape:
            obs = np.append(obs, remaining_processtime)
        if 9 in self.observation_shape:
            obs = np.append(obs, cantake)
        if 10 in self.observation_shape:
            obs = np.append(obs, isempty)
        if 11 in self.observation_shape:
            obs = np.append(obs, info)
        if 13 in self.observation_shape:
            obs = np.append(obs, full_conveyor)
        if 14 in self.observation_shape:
            obs = np.append(obs, divert_points)

        return obs

 ########################################################################################################################################################
 ## RESET FUNCTION 
 #            
    def reset(self):
        """reset all the variables to zero, empty queues
        must return the current state of the environment"""
        self.run_count +=1
        self.episode +=1
        print('Ep: {:5}, steps: {:3}, R: {:3.3f}'.format(self.episode, self.steps, self.reward), end='\r')
        #print('\n {}'.format(self.termination_cases), end='\r')
        self.D_states = {}
        for i in range(1,len(self.diverter_locations)+1):
            self.D_states[i] = False
        
        #initialize output points
        self.O_states = {}

        for i in range(1,len(self.output_locations)+1):
            self.O_states[i] = 0
        
        # initialize transition points: 0=no transition, 1=transition
        self.T_states = {}
        for i in range(1,len(self.operator_locations)+1):
            self.T_states[i] = False

        #initialize merge points
        self.M_states = {}
        for i in range(1,len(self.merge_locations)+1):
            self.M_states[i] = False

####### FOR SIMULATION ONLY 
        self.W_times = {}
        for i in range(1,len(self.operator_locations)+1):
            self.W_times[i] = self.process_time_at_GTP + 8*self.amount_of_gtps  -5
        logging.debug("Process times at operator are: {}".format(self.W_times))
####### FOR SIMULATION ONLY

        #empty amount of items on conv.
        self.items_on_conv = []
        self.reward = 0
        self.total_travel = 0
        self.steps = 0
        self.terminate = False
        self.step_reward_p = 0
        self.step_reward_n = 0
        self.O_state = 0

        #reset tracers
        self.amount_of_items_on_conv = 0
        self.amount_of_items_in_sys = 0
        self.remaining_demand = 0
        self.amount_of_orders_processed = 0
        self.positive_reward = 0
        self.negative_reward = 0
        self.cycle_count = 0
        self.items_processed = 0

        self.queues = [random.choices(np.arange(1, self.amount_of_outputs + 1),
                                      [0.33, 0.34, 0.33], k=self.gtp_buffer_size) for item in
                       range(self.amount_of_gtps)]  # generate random queues
        self.init_queues = copy(self.queues)
        self.demand_queues = copy(self.queues)
        self.in_queue = [[] for item in range(len(self.queues))]
        self.empty_env = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)
        self.carrier_type_map = np.zeros((self.empty_env.shape[0],self.empty_env.shape[1],1))

        self.do_warm_start(3)

        return self.make_observation()

########################################################################################################################################################
## PROCESSING OF ORDER CARRIERS AT GTP
# 
    def process_at_GTP(self):
        # for each step; check if it needed to process an order carrier at GTP
        O_locs = deepcopy(self.operator_locations)
        for Transition_point in O_locs:                                 #For all operator locations, check:

            try:
                if self.demand_queues[O_locs.index(Transition_point)][0] != self.in_queue[O_locs.index(Transition_point)][0]:
                    self.condition_to_transfer = True
                elif self.demand_queues[O_locs.index(Transition_point)][0] == self.in_queue[O_locs.index(Transition_point)][0]:
                    self.condition_to_process = True
            except:
                self.condition_to_transfer = False
                self.condition_to_process = False

            if self.W_times[O_locs.index(Transition_point)+1] == 0:     #if the waiting time is 0:
                logging.debug('Waiting time at GTP {} is 0, check done on correctness:'.format(O_locs.index(Transition_point)+1))
                if random.random() < self.exception_occurence: #if the random occurence is below exception occurence (set in config) do:
                    #remove an order carrier (broken)
                    logging.debug('With a change percentage an order carrier is removed')
                    logging.debug('transition point is: {}'.format(Transition_point))
                    #self.update_queues(O_locs.index(Transition_point)+1, [item[1] for item in self.items_on_conv if item[0] == Transition_point][0])
                    self.W_times[O_locs.index(Transition_point)+1] = 1
                    #self.O_states[[item[1] for item in self.items_on_conv if item[0] == Transition_point][0]] +=1
                    self.items_on_conv = [item for item in self.items_on_conv if item[0] !=Transition_point]
                
                elif self.condition_to_transfer:
                    #move order carrier back onto system via transfer - merge
                    for item  in self.items_on_conv:
                        if item[0] == Transition_point:
                            item[0][0] -=1
                    self.W_times[O_locs.index(Transition_point)+1] = 1
                    self.update_queues(O_locs.index(Transition_point)+1, self.in_queue[O_locs.index(Transition_point)][0])
                elif self.condition_to_process:
                    #Process an order at GTP successfully
                    logging.debug('Demand queues : {}'.format(self.demand_queues))
                    logging.debug('In queue : {}'.format(self.in_queue))
                    logging.debug('items on conveyor : {}'.format(self.items_on_conv))
                    logging.debug('right order carrier is at GTP (location: {}'.format(Transition_point))
                    logging.debug('conveyor memory before processing: {}'.format(self.items_on_conv))
                    self.items_on_conv = [item for item in self.items_on_conv if item[0] !=Transition_point]


                    self.amount_of_orders_processed +=1
                    logging.debug('order at GTP {} processed'.format(O_locs.index(Transition_point)+1))
                    logging.debug('conveyor memory after processing: {}'.format(self.items_on_conv))

                    #when processed, remove order carrier from demand queue
                    try:
                        #remove from demand queue
                        self.demand_queues[O_locs.index(Transition_point)] = self.demand_queues[O_locs.index(Transition_point)][1:]
                    except:
                        logging.debug("Except: Demand queue for this lane is allready empty")

                    #set new timestep for the next order
                    try: 
                        next_type = [item[1] for item in self.items_on_conv if item[0] == [Transition_point[0], Transition_point[1]-1]][0]

                    except:
                        next_type = 99
                    self.W_times[O_locs.index(Transition_point)+1] = self.process_time_at_GTP if next_type == 1 else self.process_time_at_GTP+30 if next_type == 2 else self.process_time_at_GTP+60 if next_type == 3 else self.process_time_at_GTP+60 if next_type == 4 else self.process_time_at_GTP+60
                    logging.debug('new timestep set at GTP {} : {}'.format(O_locs.index(Transition_point)+1, self.W_times[O_locs.index(Transition_point)+1]))
                else:
                    logging.debug('Else statement activated')

                #remove from in_queue when W_times is 0
                try:
                    #remove item from the In_que list
                    self.in_queue[O_locs.index(Transition_point)] = self.in_queue[O_locs.index(Transition_point)][1:]
                    logging.debug('item removed from in-que')
                except:
                    logging.debug("Except: queue was already empty!")
            elif self.W_times[O_locs.index(Transition_point)+1] < 0:
                self.W_times[O_locs_locations.index(Transition_point)+1] = 0
                logging.debug("Waiting time was below 0, reset to 0")
            else:
                self.W_times[O_locs.index(Transition_point)+1] -= 1 #decrease timestep with 1
                logging.debug('waiting time decreased with 1 time instance')
                logging.debug('waiting time at GTP{} is {}'.format(O_locs.index(Transition_point)+1, self.W_times[O_locs.index(Transition_point)+1]))
            

########################################################################################################################################################
## STEP FUNCTION
#
    def step_env(self):
        logging.debug('W-times are: {}'.format(self.W_times))

####make carrier type map
        self.carrier_type_map = np.zeros((self.empty_env.shape[0],self.empty_env.shape[1],1)).astype(float)
        for item in self.items_on_conv:
            self.carrier_type_map[item[0][1]][item[0][0]] = item[1]
            item[2] += 1/self.max_time_in_system                                                       #increase the time in the system


#### Process the orders at GTP > For simulation: do incidental transfer of order carrier
        self.process_at_GTP()


####Divert when at diverter, and diverter is set to true
####Do step for all items on conv
        for item in self.items_on_conv:
            #check diverters; is any of them True and is there an order carrier? move into lane.
            if item[0] in self.diverter_locations:
                try:
                    # 1. if diverter state == true
                    logging.debug('item[0] == {}, self.diverter_locations.index(item[0]) = {}, value D_states; {}'.format(item[0], self.diverter_locations.index(item[0]), self.D_states[self.diverter_locations.index(item[0])+1]))
                    cond_1 = self.D_states[self.diverter_locations.index(item[0])+1]
                    # 2. and the queue is not full
                    cond_2 = self.carrier_type_map[item[0][1]+1][item[0][0]] ==0
                    # 3. and the current demand == type of carrier
                    cond_3 = item[1] == self.init_queues[self.diverter_locations.index(item[0])][0]
                    logging.debug('condition 1 = {}, and condition 2= {}, condition 3 = {}'.format(cond_1, cond_2, cond_3))
                    if cond_1 and cond_2 and cond_3:
                        self.in_queue[self.diverter_locations.index(item[0])].append(item[1])
                        self.remove_from_queue(self.diverter_locations.index(item[0])+1)
                        item[0][1] += 1
                        self.reward += self.positive_reward_for_divert

                    elif cond_1 and cond_2 and not cond_3:
                        item[0][0] -= 1
                        self.reward += self.negative_reward_for_wrong_demand

                    elif cond_1 and not cond_2 and cond_3:
                        item[0][0] -= 1
                        self.reward += self.negative_reward_full_queue

                    elif cond_1 and not cond_2 and not cond_3:
                        item[0][0] -= 1
                        #self.reward += self.negative_reward_full_queue + self.negative_reward_for_wrong_demand

                    elif not cond_1 and cond_2 and cond_3:
                        # self.in_queue[self.diverter_locations.index(item[0])].append(item[1])
                        # self.remove_from_queue(self.diverter_locations.index(item[0]) + 1)
                        # item[0][1] += 1
                        item[0][0] -= 1
                        self.reward -= self.negative_reward_for_wrong_demand

                    else:
                        item[0][0] -=1

                except:
                    logging.debug("Item of size {} not moved into lane: Divert value not set to true".format(item[1])) 

            #otherwise; all items set a step in their moving direction 
            elif item[0][1] == 7 and item[0][0] > 1 :#and self.carrier_type_map[item[0][1]][item[0][0]-1] ==0: #if on the lower line, and not reached left corner:
                item[0][0] -=1                     #move left
                logging.debug('item {} moved left'.format(item[0]))
            elif item[0][0] ==1 and item[0][1] >2 : #and self.carrier_type_map[item[0][1]-1][item[0][0]] ==0: #if on left lane, and not reached top left corner:
                item[0][1] -=1
                logging.debug('item {} moved up'.format(item[0]))                    #move up
            elif item[0][1] == 2 and item[0][0] < self.empty_env.shape[1]-2 :#and self.carrier_type_map[item[0][1]][item[0][0]+1] ==0: #if on the top lane, and not reached right top corner:
                item[0][0] +=1                      #Move right
                logging.debug('item {} moved right'.format(item[0]))
            elif item[0][0] == self.empty_env.shape[1]-2 and item[0][1] <7 : #and self.carrier_type_map[item[0][1]+1][item[0][0]] ==0: #if on right lane, and not reached right down corner:
                item[0][1] +=1
                logging.debug('item {} moved down'.format(item[0]))
            elif item[0][1] > 7 and item[0][0] in [lane[0] for lane in self.diverter_locations] and item[0][1] < self.empty_env.shape[0]-1 and item[0][0] < self.amount_of_gtps*4+3 and self.carrier_type_map[item[0][1]+1][item[0][0]] ==0: #move down into lane
                item[0][1] +=1
                logging.debug('item {} moved into lane'.format(item[0]))
            elif item[0][1] > 7 and item[0][0] in [lane[0] for lane in self.merge_locations] and item[0][0] < self.amount_of_gtps*4+3 and self.carrier_type_map[item[0][1]-1][item[0][0]] ==0 and self.carrier_type_map[item[0][1]-1][item[0][0]+1] ==0: #move up into merge lane
                item[0][1] -=1
            elif item[0][1] > 7 and item[0][0] > self.amount_of_gtps*4+3 and self.carrier_type_map[item[0][1]-1][item[0][0]] ==0: #move up if on output lane
                item[0][1] -=1
                logging.debug('item {} moved onto conveyor'.format(item[0]))

        OL = deepcopy(self.output_locations)
        ####try to add new item from output when On!=0
        if self.O_state == 0:
            pass #dont do output
        if self.O_state == 1 and self.carrier_type_map[OL[0][1]][OL[0][0]+1] ==0:
            self.items_on_conv.append([OL[0], 1, 0])
        elif self.O_state == 1 and self.carrier_type_map[OL[0][1]][OL[0][0]+1] !=0:
            self.reward += self.negative_reward_for_invalid

        elif self.O_state == 2 and self.carrier_type_map[OL[1][1]][OL[1][0]+1] ==0:
            self.items_on_conv.append([OL[1], 2, 0])
        elif self.O_state == 1 and self.carrier_type_map[OL[1][1]][OL[1][0]+1] !=0:
            self.reward += self.negative_reward_for_invalid

        elif self.O_state == 3 and self.carrier_type_map[OL[2][1]][OL[2][0]+1] ==0:
            self.items_on_conv.append([OL[2], 3, 0])
        elif self.carrier_type_map[OL[2][1]][OL[2][0]+1] !=0:
            self.reward += self.negative_reward_for_invalid


    def do_step(self, actions):
        action1, action2, action3, action4 = actions
        if action1==0:
            self.D_states[1] = False
        elif action1 ==1:
            self.D_states[1] = True

        if action2==0:
            self.D_states[2] = False
        elif action2 ==1:
            self.D_states[2] = True

        if action3==0:
            self.D_states[3] = False
        elif action3 ==1:
            self.D_states[3] = True

        if action4 == 0:
            self.O_state = 0
        elif action4 == 1:
            self.O_state = 1
        elif action4 == 2:
            self.O_state = 2
        elif action4 == 3:
            self.O_state = 3

        self.step_env()

    def step(self,action):
        self.reward = 0
        self.steps += 1

        self.do_step(self.actions_map[action])

        # rewards for taking cycles in the system
        if len([item for item in self.items_on_conv if
                item[0] == [1, 7]]) == 1:  # in case that negative reward is calculated with cycles
            self.reward += self.negative_reward_for_cycle  # punish if order carriers take a cycle #tag:punishment
            self.cycle_count += 1

        # Determine termination cases
        if self.termination_condition == 1:
            if self.demand_queues == [[] * i for i in range(self.amount_of_gtps)]:
                self.terminate = True
                self.termination_cases['demand_satisfied'] +=1
        elif self.termination_condition == 2:
            if self.init_queues == [[] * i for i in range(self.amount_of_gtps)]:
                self.terminate = True
                self.termination_cases['queues_satisfied'] += 1
        elif self.termination_condition ==3:
            if self.amount_of_orders_processed > self.amount_to_process:
                self.terminate = True
                self.termination_cases['amounts_processed'] += 1

        # Terminate for too much steps
        if self.steps > 10000:
            self.reward += self.negative_reward_for_invalid
            self.terminate = True
            self.termination_cases['too_much_steps'] += 1

        #terminate for too much similar steps
        if self.same_action_count > 500:
            self.terminate = True
            self.termination_cases['too_much_sim_steps'] += 1

        #terminate when too many cycles
        if self.cycle_count > self.max_cycle_count:
            self.reward += self.negative_reward_for_invalid
            self.terminate = True
            self.termination_cases['cycle_count'] += 1

        next_state = self.make_observation()
        reward = self.reward
        terminate = self.terminate
        return next_state, reward, terminate, {}

   

################## RENDER FUNCTIONS ################################################################################################
    def render_plt(self):
        """Simple render function, uses matplotlib to render the image + some additional information on the transition points"""
        # print('items on conveyor:')
        # print(self.items_on_conv)
        # print('states of Divert points = {}'.format(self.D_states))
        # print('states of Output points = {}'.format(self.O_states))
        # for queue in self.init_queues:
            # print('Queue GTP{}: {}'.format(self.init_queues.index(queue), queue))

        self.image = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)

        for item in self.items_on_conv:
            self.image[item[0][1]][item[0][0]] = np.asarray([self.pallette[0] if item[1] ==1 else self.pallette[1] if item[1] ==2 else self.pallette[2] if item[1] ==3 else self.pallette[3]]) 
        self.image = self.image / 255.0
        plt.imshow(np.asarray(self.image))
        plt.show()
    

    def render1(self):
        """render with opencv, for faster processing"""
        #printable check
        printable_check = True
        if printable_check == True:
            for item in self.items_on_conv:
                self.carrier_type_map[item[0][1]][item[0][0]] = item[1]
            print(self.carrier_type_map)


        resize_factor = 35
        image = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)

        for item in self.items_on_conv:
            image[item[0][1]][item[0][0]] = np.asarray([self.pallette[0] if item[1] ==1 else self.pallette[1] if item[1] ==2 else self.pallette[2] if item[1] ==3 else self.pallette[3]]) 

        #resize with PIL
        im = Image.fromarray(np.uint8(image))
        img = im.resize((image.shape[1]*resize_factor,image.shape[0]*resize_factor), resample=Image.BOX) #BOX for no anti-aliasing)
        cv2.imshow("Simulation-v0.9", cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)

    def render(self):
        """render with opencv, for faster processing"""
        resize_factor = 36
        box_diameter = 30
        self.image = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)
        im = Image.fromarray(np.uint8(self.image))
        img = im.resize((self.image.shape[1]*resize_factor,self.image.shape[0]*resize_factor), resample=Image.BOX) #BOX for no anti-aliasing)
        draw = ImageDraw.Draw(img)

        for i in range(7):
            for item in copy(self.output_locations):
                x0 = item[0] * resize_factor + 3
                y0 = item[1] * resize_factor + 40 + 3 + i * 35
                box_size = 20 if item ==self.output_locations[0] else 25 if item ==self.output_locations[1] else 30 if item ==self.output_locations[2] else 30
                x1 = x0 + box_size
                y1 = y0 + box_size
                color = self.pallette[0] if item ==self.output_locations[0] else self.pallette[1] if item ==self.output_locations[1] else self.pallette[2] if item ==self.output_locations[2] else self.pallette[2]
                draw.rectangle([x0,y0,x1,y1], fill=tuple(color), outline='black')
        
        #Draw the order carriers
        for item in self.items_on_conv:
            size = box_diameter-10 if item[1]==1 else box_diameter-5 if item[1]==2 else box_diameter if item[1]==3 else box_diameter
            x0 = item[0][0] * resize_factor +3
            x1 = x0 + size
            y0 = item[0][1] * resize_factor + 3
            y1 = y0 + size
            color = self.pallette[0] if item[1] ==1 else self.pallette[1] if item[1] ==2 else self.pallette[2] if item[1] ==3 else self.pallette[3]            
            draw.rectangle([x0, y0, x1, y1], fill=tuple(color), outline='black')


        #Draw demands
        for item in copy(self.diverter_locations):
            x0 = item[0] * resize_factor+ 40
            y0 = item[1] * resize_factor+ 40
            x1 = x0 + 30
            y1 = y0 + 30
            
            try: next_up = self.init_queues[self.diverter_locations.index(item)][0]
            except: next_up = '-'
            color = self.pallette[0] if next_up ==1 else self.pallette[1] if next_up ==2 else self.pallette[2] if next_up ==3 else (225,225,225)
            draw.ellipse([x0, y0, x1, y1], fill=tuple(color), outline=None)
            draw.text((x0 + 10, y0 + 5), '{}'.format(next_up), fill= 'black', font=ImageFont.truetype(font='arial', size=20, index=0, encoding='unic', layout_engine=None))
            draw.text((x0, y1+ 5), 'Demand \n Queue', font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))

            #draw demand conditions
            x2, y2 = item[0] * resize_factor, item[1] * resize_factor- 12
            x3, y3 = x2 + 10, y2 + 10
            x4, y4 = x3 + 5, y2
            x5, y5 = x4 + 10, y4 + 10
            color1 = 'green' if self.D_condition_1[self.diverter_locations.index(item)+1] == True else 'red'
            color2 = 'green' if self.D_condition_2[self.diverter_locations.index(item)+1] == True else 'red'
            draw.ellipse([x2, y2, x3, y3], fill=color1, outline=None)
            draw.ellipse([x4, y4, x5, y5], fill=color2, outline=None)

            #init queues on top
            x6, y6 = item[0] * resize_factor - 30, item[1] * resize_factor - 30
            draw.text((x6 + 10, y6 + 5), '{}'.format(self.init_queues[self.diverter_locations.index(item)][:5]), fill= 'white', font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))
            
            #in_queue
            x7, y7 = x0, y0 + 95
            draw.text((x7, y7), 'In queue', fill= 'white', font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))
            draw.text((x7, y7+ 15), '{}'.format(self.in_queue[self.diverter_locations.index(item)][:5]), fill= 'white', font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))

##### TURN OFF FOR FASTER RENDERING #############################################################################################
        #values of the O_states
        for item in copy(self.output_locations):
                x0 = item[0] * resize_factor + 40
                y0 = item[1] * resize_factor + 40
                draw.text((x0, y0), '{}'.format(self.O_states[self.output_locations.index(item)+1]), fill='white',
                          font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))

        #draw reward
        x0, y0 = self.diverter_locations[0][0] * resize_factor + 130, self.diverter_locations[0][1] * resize_factor + 150
        y1 = y0 + 25
        y2 = y1 + 25
        draw.text((x0, y0), ' Total Reward: {}'.format(self.reward), fill='white',
                      font=ImageFont.truetype(font='arial', size=15, index=0, encoding='unic', layout_engine=None))
        draw.text((x0, y1), ' Positive Reward: {}'.format(self.step_reward_p), fill='green',
                      font=ImageFont.truetype(font='arial', size=15, index=0, encoding='unic', layout_engine=None))
        draw.text((x0, y2), ' Negative Reward: {}'.format(self.step_reward_n), fill='red',
                      font=ImageFont.truetype(font='arial', size=15, index=0, encoding='unic', layout_engine=None))
###################################################################################################################################

        #Draw GTP demands
        for item in copy(self.operator_locations):
            x0 = item[0] * resize_factor +40
            y0 = item[1] * resize_factor
            x1 = x0 + 30
            y1 = y0 + 30
            
            try: next_up = self.demand_queues[self.operator_locations.index(item)][0]
            except: next_up = '-'
            color = self.pallette[0] if next_up ==1 else self.pallette[1] if next_up ==2 else self.pallette[2] if next_up ==3 else (225,225,225)
            draw.ellipse([x0, y0, x1, y1], fill=tuple(color), outline=None)
            draw.text((x0 + 10, y0 + 5), '{}'.format(next_up), fill= 'black', font=ImageFont.truetype(font='arial', size=20, index=0, encoding='unic', layout_engine=None))
            draw.text((x0, y0 -45), 'Demand \n at GtP', font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))

            #demand queues
            draw.text((x0, y0 -15), '{}'.format(self.demand_queues[self.operator_locations.index(item)][:5]), font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))

        #resize with PIL
        #img = img.resize((1200,480), resample=Image.BOX)
        cv2.imshow(self.window_name, cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    def create_window(self):
        # cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(self.window_name, 1200, 480)
        pass


    def run(self, model, episodes=1000):
        """
        Use a trained model to select actions

        """
        try:
            for episode in range(episodes):
                self.done, step = False, 0
                state = self.reset()
                while not self.done:
                    action = model.model.predict(state)
                    state, reward, self.done, _ = self.step(action[0])
                    print(
                        '   Episode {:2}, Step {:3}, Reward: {:.2f}, State: {}, Action: {:2}'.format(episode, step, reward,
                                                                                                     state[0], action[0]),
                        end='\r')
                    self.render()
                    step += 1
        except KeyboardInterrupt:
            pass

    def sample(self):
        """
        Sample random actions and run the environment
        """
        self.create_window()
        for _ in range(10):
            self.done = False
            state = self.reset()
            while not self.done:
                action = self.action_space.sample()
                state, reward, self.terminate, _ = self.step(action)
                print('Reward: {:2.3f}, state: {}, action: {}'.format(reward, state, action))
                self.render()
        cv2.destroyAllWindows()

from rl.baselines import get_parameters

#env = simple_conveyor_2(get_parameters('simple_conveyor_2'))