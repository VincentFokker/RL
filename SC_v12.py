##############################################################################
# Version: 2.0                                                               #
# log: second version to run in the wrapper                                   #
# Includes an adjusted reward function and state obs                          #
# run with > python train.py -e simple_conveyor_1 -s Test -n test123         #
# TEST with > python test.py -e simple_conveyor -s PPO2 -n 0 --render

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
from copy import copy
from random import randint
import seaborn as sns
import random
import logging
import gym
import time


#CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements
logging.basicConfig(level=logging.INFO)

class simple_conveyor_2(gym.Env):

######## INITIALIZATION OF VARIABLES ###############################################################################################################
    def __init__(self, config, queues, **kwargs):
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



        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(4)

        #determination of observation_space:
        # (2*width+width+2*height+height) * 2 + (10*amount_of_gtp*2(binary)*2(for init and queue) = shapesize
        # (2*(self.amount_of_gtps *4 + 13) + (self.amount_of_gtps *4 + 13) + 2*4 + 4) * 2 + (10 * (self.amount_of_gtps *4 + 13) * 2 * 2)
        # (50+25+8+4)                     * 2 + (10*3*2*2) = 174 + 120 = 294 shapesize
        self.shape = 2*((2*((self.amount_of_gtps*4) + 13)) + ((self.amount_of_gtps *4) + 13) + ((2*4 + 4))) + (10 * self.amount_of_gtps * 2 * 2)
        self.observation_space = gym.spaces.Box(shape=(self.shape, ),
                                                high=self.max_time_in_system, low=0,
                                                dtype=np.uint8)

        #init queues
        self.queues = queues
        logging.info("queues that are initialized: {}".format(self.queues))
        self.init_queues = copy(self.queues)
        self.demand_queues = copy(self.queues)
        self.in_queue = [[] for item in range(len(self.queues))]

        ## rewards
        # All punishments found by: tag:punishment
        self.negative_reward_per_step = self.config['negative_reward_per_step']
        self.travelpath_to_gtp_reward = self.config['travelpath_to_gtp_reward']
        self.negative_reward_for_cycle = self.config['negative_reward_for_cycle']
        self.negative_reward_for_flooding = self.config['negative_reward_for_flooding']
        self.negative_reward_for_empty_queue = self.config['negative_reward_for_empty_queue']
        self.positive_reward_for_divert = self.config['positive_reward_for_divert']

        #tracers
        self.amount_of_items_on_conv = 0
        self.amount_of_items_in_sys = 0
        self.remaining_demand = 0
        self.amount_of_orders_processed = 0
        self.positive_reward = 0
        self.negative_reward = 0

        #gym related part
        self.reward = 0
        self.total_travel = 0
        self.terminate = False
        self.episode= 0
        self.steps = 0

        self.create_window()

        #colors
        self.pallette = (np.asarray(sns.color_palette("Reds", self.amount_of_outputs)) * 255).astype(int)

        # build env
        self.empty_env = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)
        self.image = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)

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

###### HELPER FUNCTIONS ############################################################################################################################
    def get_candidate_lists(self, list_lists, x):
        """Returns all lists, starting with x in list of lists"""
        return [nestedlist for nestedlist in list_lists if nestedlist[0] == x]

    def len_longest_sublist(self, listoflists):
        """returns length of the longest list in the sublists"""
        try:
            return max([len(sublist) for sublist in listoflists]) 
        except:
            return 0

    def len_shortest_sublist(self, listoflists):
        """returns length of the shortest list in the sublists"""
        return min([len(sublist) for sublist in listoflists])
        
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

    
    def simulate_operator_action(self):
        'processes an item at all the GTP stations, currently just accepts the item allways'
        self.items_on_conv = [sublist for sublist in self.items_on_conv if sublist[0] not in self.operator_locations]
        
    def encode(self, var):
        """encodes categorical variables 0-3 to binary"""
        return (0,0) if var == 0 else (0,1) if var == 1 else (1,0) if var == 2 else (1,1) if var ==3 else var
#####################################################################################
## Make Observation
#
    def make_observation(self):
        '''Builds the observation from the available variables'''

        ### For the obeservation of the conveyor ########################################################################
        self.carrier_type_map_obs = np.zeros((self.empty_env.shape[0], self.empty_env.shape[1], 1)).astype(float)
        self.carrier_type_map_obs1 = np.zeros((self.empty_env.shape[0], self.empty_env.shape[1], 1)).astype(float)

        for item in self.items_on_conv:
            self.carrier_type_map_obs[item[0][1]][item[0][0]] = item[1]
            self.carrier_type_map_obs1[item[0][1]][item[0][0]] = item[2]
        # cut padding
        type_map_obs = self.carrier_type_map_obs[2:8, 1:-1]                     #for the carrier type
        type_map_obs1 = self.carrier_type_map_obs1[2:8, 1:-1]                   #for the time in the system
        # row 0 and -1 (top and bottom)
        conv_top_bottom = np.append(type_map_obs[0], type_map_obs[-1])          #top and bottom lane for the carrier type
        logging.debug('topbottom lenght = {}'.format(len(conv_top_bottom)))
        conv_top_bottom1 = np.append(type_map_obs1[0], type_map_obs1[-1])       #top and bottom lane for the time in the system
        # left and right lane
        conv_left_right = np.append(type_map_obs[1:-1][:, 0], type_map_obs[1:-1][:, -1])        #left and right lane for carrier type
        conv_left_right1 = np.append(type_map_obs1[1:-1][:, 0], type_map_obs1[1:-1][:, -1])     #left and right for the time in system
        logging.debug('leftright lenght = {}'.format(len(conv_left_right)))
        # together
        carrier_type_map_obs = np.append(conv_top_bottom, conv_left_right)                      #full circle for the carrier type
        # type_map_obs = np.append(type_map_obs[-1], type_map_obs[1:-1][:, -1])            #for half observation
        type_map_obs = np.array([self.encode(item) for item in list(carrier_type_map_obs)]).flatten()   #binary encoded memory for the type
        logging.debug('typemap lenght = {}'.format(len(type_map_obs)))

        type_map_obs1 = np.append(conv_top_bottom1, conv_left_right1)                   #full circle for the time in system
        # type_map_obs1 = np.append(type_map_obs1[-1], type_map_obs1[1:-1][:, -1])          #for half observation
        logging.debug('time in system lenght = {}'.format(len(type_map_obs1)))

        #combine the type and the time in system
        type_map_obs = np.append(type_map_obs, type_map_obs1)
        logging.debug('size conveyor obs = {}'.format(len(type_map_obs)))

        ### For the observation of the items in queue ##################################################################
        in_queue = []
        for item in self.in_queue:
            in_queue.append(item + [0] * (10 - len(item)))
        in_queue = np.array(in_queue).flatten()
        # binary encoding of the categorical variables
        in_queue = np.array([self.encode(item) for item in list(in_queue)]).flatten()
        logging.debug('in_queue lenght = {}'.format(len(in_queue)))

        ### For the observation of the demand of the GtP Queue #########################################################
        #make the init list
        init = []
        for item in self.init_queues:
            init1 = item[:10]
            init.append(init1 + [0] * (10 - len(init1)))
        init = list(np.array(init).flatten())
        # binary encoding of the categorical variables
        init = np.array([self.encode(item) for item in init]).flatten()
        logging.debug('init lenght = {}'.format(len(init)))

        ### Combine All to one array ###################################################################################

        obs = np.append(np.array(init).flatten(), type_map_obs)         #combine GTP queue with the conveyor memory
        obs = np.append(in_queue, obs)                                  #add the information about what is in queue
        logging.debug('size of observation is: {}'.format(len(obs)))
        return obs

 ########################################################################################################################################################
 ## RESET FUNCTION 
 #            
    def reset(self):
        """reset all the variables to zero, empty queues
        must return the current state of the environment"""

        self.episode +=1
        print('Ep: {:5}, steps: {:3}, R: {:3.3f}'.format(self.episode, self.steps, self.reward), end='\r')
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
            self.W_times[i] = self.process_time_at_GTP + 8*self.amount_of_gtps  + randint(-10, 10)
        logging.debug("Process times at operator are: {}".format(self.W_times))
####### FOR SIMULATION ONLY

        #empty amount of items on conv.
        self.items_on_conv = []
        self.reward = 0
        self.total_travel = 0
        self.steps = 0
        self.terminate = False

        #reset tracers
        self.amount_of_items_on_conv = 0
        self.amount_of_items_in_sys = 0
        self.remaining_demand = 0
        self.amount_of_orders_processed = 0
        self.positive_reward = 0
        self.negative_reward = 0

        # self.queues = [random.choices(np.arange(1, self.amount_of_outputs + 1),
        #                               [self.percentage_small_carriers, self.percentage_medium_carriers,
        #                                self.percentage_large_carriers], k=self.gtp_buffer_size) for item in
        #                range(self.amount_of_gtps)]  # generate random queues
        self.init_queues = copy(self.queues)
        self.demand_queues = copy(self.queues)
        self.in_queue = [[] for item in range(len(self.queues))]
        self.empty_env = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)
        self.carrier_type_map = np.zeros((self.empty_env.shape[0],self.empty_env.shape[1],1))

        return self.make_observation()

########################################################################################################################################################
## PROCESSING OF ORDER CARRIERS AT GTP
# 
    def process_at_GTP(self):
        # for each step; check if it needed to process an order carrier at GTP
        O_locs = copy(self.operator_locations)
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
                    logging.info('trasition point is: {}'.format(Transition_point))
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
                    self.reward += 10 + 10 + (O_locs.index(Transition_point)+1 * 4) * self.travelpath_to_gtp_reward
                    self.total_travel += 10 + 10 + (O_locs.index(Transition_point)+1 * 4)
                    self.positive_reward += 10 + 10 + (O_locs.index(Transition_point)+1 * 4) * self.travelpath_to_gtp_reward
                    self.amount_of_orders_processed +=1
                    logging.debug('order at GTP {} processed'.format(O_locs.index(Transition_point)+1))
                    logging.debug('conveyor memory after processing: {}'.format(self.items_on_conv))

                    #when processed, remove order carrier from demand queue
                    try:
                        #remove from demand queue
                        self.demand_queues[O_locs.index(Transition_point)] = self.demand_queues[O_locs.index(Transition_point)][1:]
                    except:
                        logging.info("Except: Demand queue for this lane is allready empty")

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

####make carrier type map
        self.carrier_type_map = np.zeros((self.empty_env.shape[0],self.empty_env.shape[1],1)).astype(float)
        for item in self.items_on_conv:
            self.carrier_type_map[item[0][1]][item[0][0]] = item[1]
            item[2] += 1/self.max_time_in_system                                                       #increase the time in the system

        # give a negative reward for each step, for all the items that are taking are taking a loop
        # self.reward += len([item for item in self.items_on_conv if item[0][1] < 7]) * self.negative_reward_per_step       #tag:punishment

#### Process the orders at GTP > For simulation: do incidental transfer of order carrier
        self.process_at_GTP()

####toggle diverters if needed
        #toggle All D_states if needed:
        d_locs = copy(self.diverter_locations)
        carrier_map = copy(self.carrier_type_map)
        for loc2 in d_locs:
            try:
                #Condition 1 = if the carrier type at any of the diverter locations is EQUAL TO the next-up requested carrier type at GTP request lane of this specific diverter location
                condition_1 = carrier_map[loc2[1]][loc2[0]] == self.init_queues[d_locs.index(loc2)][0]
                if condition_1 == True:
                    self.D_condition_1[d_locs.index(loc2)+1] = True
                else:
                    self.D_condition_1[d_locs.index(loc2)+1] = False
                #condition 2 = if the lenght of the in_queue is <= smallest queue that also demands order carrier of the same type
                condition_2 = len(self.in_queue[d_locs.index(loc2)])-1 <= min(map(len, self.in_queue))
                if condition_2 == True:
                    self.D_condition_2[d_locs.index(loc2)+1] = True
                else:
                    self.D_condition_2[d_locs.index(loc2)+1] = False

                condition_3 = carrier_map[loc2[1]+1][loc2[0]] ==0
                logging.debug(carrier_map[loc2[1]+1][loc2[0]] ==0)


                if condition_1 and condition_2 and condition_3: 
                    self.D_states[d_locs.index(loc2)+1] = True
                    logging.debug("set diverter state for diverter {} to TRUE".format(d_locs.index(loc2)+1))
                    self.remove_from_queue(d_locs.index(loc2)+1)
                    logging.debug("request removed from demand queue")
                    self.add_to_in_que(d_locs.index(loc2),int(carrier_map[loc2[1]][loc2[0]]))
                    logging.debug("Order carrier added to GTP queue")

                else:
                    self.D_states[d_locs.index(loc2)+1] = False
                    logging.debug("Divert-set requirement not met at cord {}".format(loc2))
            except IndexError:
                logging.debug('Index error: queues are empty!')
                self.D_states[d_locs.index(loc2)+1] = False
            except:
                logging.warning('Another error occurred; this should not happen! Investigate the cause!')
                self.D_states[d_locs.index(loc2)+1] = False



        

####Divert when at diverter, and diverter is set to true
####Do step for all items on conv
        for item in self.items_on_conv:
            #check diverters; is any of them True and is there an order carrier? move into lane.
            if item[0] in self.diverter_locations:
                try:
                    if self.D_states[self.diverter_locations.index(item[0])+1] == True and self.carrier_type_map[item[0][1]+1][item[0][0]] ==0:
                        item[0][1] +=1
                        self.reward += self.positive_reward_for_divert
                        self.positive_reward += self.positive_reward_for_divert
                        logging.debug("moved order carrier into GTP lane {}".format(self.diverter_locations.index(loc1)+1))
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

        ####try to add new item from output when On!=0
        for cord2 in self.output_locations:
            loc =copy(cord2)
            if self.O_states[self.output_locations.index(loc)+1] !=0 and self.carrier_type_map[cord2[1]][cord2[0]+1] ==0:
                self.items_on_conv.append([loc,self.output_locations.index(loc)+1,0])
                self.O_states[self.output_locations.index(loc)+1] -=1
                logging.debug("Order carrier outputted at {}".format(loc))
                logging.debug("Items on conveyor: {}".format(self.items_on_conv))
            else:
                logging.debug('No order carrier output on output {} .'.format(loc))


    def step(self, action):
        logging.debug("Executed action: {}".format(action))
        logging.debug("Demand queue: {}".format(self.demand_queues))
        self.steps +=1
        if action==0:
            self.step_env()
            logging.debug("- - action 0 executed")
            logging.debug("Divert locations :{}".format(self.diverter_locations))
            logging.debug('states of Divert points = {}'.format(self.D_states))
        elif action ==1: 
            self.O_states[1] +=1
            self.step_env()
            logging.debug("- - action 1 executed")
        elif action ==2:
            self.O_states[2] +=1
            self.step_env()
            logging.debug("- - action 2 executed")
        elif action ==3:
            self.O_states[3] +=1
            self.step_env()
            logging.debug("- - action 3 executed")
        else:
            #self.reward -=20                                                                                            #tag:punishment
            self.terminate = True

        logging.debug("states of O: {}".format(self.O_states))
        logging.debug("init queues :{}".format(self.init_queues))
        logging.debug('amount of items in init queues: {}'.format(sum([len(item) for item in self.init_queues])))
        logging.debug("conveyor memory : {}".format( self.items_on_conv))
        logging.debug('Amount of items on conv: {}'.format(len(self.items_on_conv)))
        logging.debug('Demand queue : {}'.format(self.demand_queues))
        logging.debug('amount of demand still needing processing: {}'.format(sum([len(item) for item in self.demand_queues])))
        logging.debug('In queue : {}'.format(self.in_queue))
        logging.debug('In queue items : {} '.format(sum([len(item) for item in self.in_queue])))
        logging.debug('In queue according to convmap : {}'.format(len([item for item in self.items_on_conv if item[0][1] > 7])))
        logging.debug('')
        logging.debug('--------------------------------------------------------------------------------------------------------------------')

        next_state = self.make_observation()

        ### calculate conditional reward ##############################################################################
        logging.info(len([item for item in self.items_on_conv if item[0] ==[1,7]]))
        if len([item for item in self.items_on_conv if item[0] ==[1,7]]) == 1:              # in case that negative reward is calculated with cycles
            self.reward += self.negative_reward_for_cycle                                   # punish if order carriers take a cycle #tag:punishment
            #self.negative_reward += self.negative_reward_for_cycle

        # if len([item for item in self.items_on_conv if item[0][1] < 8]) > len([item for sublist in self.init_queues for item in sublist]):
        #     self.reward += self.negative_reward_for_flooding                                                            #tag:punishment
        for i in range(1, self.amount_of_outputs + 1):
            if len([item[1] for item in self.items_on_conv if item[0][1] < 8 and item[1] == i]) > len(
                    [item for sublist in self.init_queues for item in sublist if item == i]):
                logging.debug('Too many items of type {} on conv! - punished!'.format(i))
                self.reward += self.negative_reward_for_flooding
                #self.negative_reward += self.negative_reward_for_flooding
            else:
                logging.debug('Not too many items of type {} on conv!'.format(i))

        for queue in self.in_queue:
            if len(queue) ==0:
                self.reward += self.negative_reward_for_empty_queue                                                     #tag:punishment
                #self.negative_reward += self.negative_reward_for_empty_queue

        ### Define some tracers     ##################################################################################
        self.amount_of_items_on_conv = len([item for item in self.items_on_conv if item[0][1] < 8])
        self.amount_of_items_in_sys = len(self.items_on_conv)
        self.remaining_demand = len([item for sublist in self.demand_queues for item in sublist])

        ### Determine Termination cases ###############################################################################
        try:
            if max([item[2] for item in self.items_on_conv]) >= 1:
                self.terminate = True
        except:
            self.terminate = False
        
        #terminate if the demand queues are empty (means all is processed)
        if self.demand_queues == [[] * i for i in range(self.amount_of_gtps)] :
            self.terminate= True

        ### Summarized return for step function ####################################################################
        reward = self.reward
        terminate = self.terminate
        logging.debug('Reward is: {}'.format(self.reward))
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
        cv2.waitKey(0)

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

# from rl.baselines import get_parameters
#
# env = simple_conveyor_2(get_parameters('simple_conveyor_2'))

############### MAIN ##############################################################################################################################################

# env = simple_conveyor_2(get_parameters('simple_conveyor_2'))
# env.reset()
# print(env.queues)
#
# #Build action list according to FIFO and Round-Robin Policy
# order_list = []
# for index in range(len(env.queues[0])):
#     order_list.append([item[index] for item in env.queues])
#
# #flat_list = [item for sublist in l for item in sublist]
# order_list = [item for sublist in order_list for item in sublist]
# logging.debug("Resulting in sequence of actions: {}".format(order_list))
#
# start = time.time()
#
# # Use order list based on Round_robin
# for item in order_list:
#     env.step(item)
#     #env.render()
#
# while env.demand_queues != [[] * i for i in range(env.amount_of_gtps)]:
#     env.step(0)
#     #env.render()
#
#
# stop = time.time()
# print('running time: {}'.format(stop-start))
# print('reward: ', env.reward)