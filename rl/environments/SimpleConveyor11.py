########################################################################################
# Version: 8.0                                                                         #
#                                           #
# Decreased observation space to (42, 0)            #
# Added action tracer
# run with > python train.py -e simple_conveyor_4 -s Test -n Test123        #
# TEST with > python test.py -e simple_conveyor_4 -s Test -n 0 --render

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
from copy import copy
import seaborn as sns
import random
import logging
import gym
import math

# CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements
logging.basicConfig(level=logging.INFO)


class SimpleConveyor11(gym.Env):

    ######## INITIALIZATION OF VARIABLES ###############################################################################################################
    def __init__(self, config):
        """initialize states of the variables, the lists used"""
        metadata = {'render.modes': ['human']}
        # init configs: for explaination of these variables, check: config/simple_conveyor_9.yml
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

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(4)

        # determination of observation_space:
        # shape = (for all 3 types of order carrier the total amount (normalized) * Amount of GTP (for each gtP) + (amount of observed instances * amount of gtp * 2(binary coding) + for each gtp_queue 2 values how full, and remaining time (normalized)
        self.shape = self.amount_of_gtps + 3 * self.amount_of_gtps + (
                    self.in_que_observed * self.amount_of_gtps * 2) + (2 * self.amount_of_gtps)
        self.observation_space = gym.spaces.Box(shape=(self.shape,),
                                                high=1, low=0,
                                                dtype=np.uint8)

        # init queues
        self.queues = [random.choices(np.arange(1, self.amount_of_outputs + 1),
                                      [self.percentage_small_carriers, self.percentage_medium_carriers,
                                       self.percentage_large_carriers],
                                      k=self.gtp_buffer_size) for item in
                       range(self.amount_of_gtps)]  # generate random queues based on distribution defined in config
        logging.info("queues that are initialized: {}".format(self.queues))

        # make required copies
        self.init_queues = copy(self.queues)
        self.demand_queues = copy(self.queues)
        self.in_queue = [[] for item in range(len(self.queues))]

        ## rewards
        # All rewards are define in the config file
        self.negative_reward_per_step = self.config['negative_reward_per_step']
        self.travelpath_to_gtp_reward = self.config['travelpath_to_gtp_reward']
        self.negative_reward_for_cycle = self.config['negative_reward_for_cycle']
        self.negative_reward_for_flooding = self.config['negative_reward_for_flooding']
        self.negative_reward_for_empty_queue = self.config['negative_reward_for_empty_queue']
        self.positive_reward_for_divert = self.config['positive_reward_for_divert']
        self.negative_reward_for_invalid = self.config['negative_reward_for_invalid']
        self.max_output_reward = self.config['max_output_reward']

        # tracers for logging during training
        self.amount_of_items_on_conv = 0
        self.amount_of_items_in_sys = 0
        self.remaining_demand = 0
        self.amount_of_orders_processed = 0
        self.positive_reward = 0
        self.negative_reward = 0
        self.cycle_count = 0
        self.run_count = 0
        self.previous_action = 0
        self.same_action_count = 0

        # gym related part
        self.reward = 0
        self.step_reward_p = 0
        self.step_reward_n = 0
        self.total_travel = 0
        self.terminate = False
        self.episode = 0
        self.steps = 0

        self.create_window()

        # colors used in the render
        self.pallette = (np.asarray(sns.color_palette("Reds", self.amount_of_outputs)) * 255).astype(int)

        # build env
        self.empty_env = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)
        self.image = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)

        # define where the operators, diverts and outputs of the GTP stations are
        self.operator_locations = [[i, self.empty_env.shape[0] - 1] for i in range(4, self.amount_of_gtps * 4 + 1, 4)][
                                  ::-1]
        self.output_locations = [[i, 7] for i in range(self.empty_env.shape[1] - self.amount_of_outputs * 2 - 1,
                                                       self.empty_env.shape[1] - 2, 2)]
        self.diverter_locations = [[i, 7] for i in range(4, self.amount_of_gtps * 4 + 1, 4)][::-1]
        self.merge_locations = [[i - 1, 7] for i in range(4, self.amount_of_gtps * 4 + 1, 4)][::-1]
        logging.debug("operator locations: {}".format(self.operator_locations))
        logging.debug("output locations: {}".format(self.output_locations))
        logging.debug("diverter locations: {}".format(self.diverter_locations))
        logging.debug("Merge locations: {}".format(self.merge_locations))

        # initialize divert points: False=no diversion, True=diversion
        self.D_states = {}
        for i in range(1, len(self.diverter_locations) + 1):
            self.D_states[i] = False

        # D conditions: for the render
        self.D_condition_1 = {}
        for i in range(1, len(self.diverter_locations) + 1):
            self.D_condition_1[i] = False

        self.D_condition_2 = {}
        for i in range(1, len(self.diverter_locations) + 1):
            self.D_condition_2[i] = False

        # initialize output points
        self.O_states = {}
        for i in range(1, len(self.output_locations) + 1):
            self.O_states[i] = 0

        # initialize transition points: 0=no transition, 1=transition (for future use)
        self.T_states = {}
        for i in range(1, len(self.operator_locations) + 1):
            self.T_states[i] = False

        # initialize merge points    (for future use)
        self.M_states = {}
        for i in range(1, len(self.merge_locations) + 1):
            self.M_states[i] = False

        ####### FOR SIMULATION ONLY
        # a counter to record the processing time at GtP station
        self.W_times = {}
        for i in range(1, len(self.operator_locations) + 1):
            self.W_times[
                i] = self.process_time_at_GTP + 8 * self.amount_of_gtps - 5  # initialize with some time for first run
        logging.debug("Process times at operator are:{}".format(self.W_times))

        ####### FOR SIMULATION ONLY
        self.condition_to_transfer = False
        self.condition_to_process = False

        # initialize conveyor memory
        self.items_on_conv = []
        self.carrier_type_map = np.zeros((self.empty_env.shape[0], self.empty_env.shape[1], 1))

        ######## Do a warm_start
        self.warm_start()

    def warm_start(self):
        # add items to queues, so queues are not empty when starting with training (empty queue is punished with -1 each timestep)
        for _ in self.operator_locations:
            self.in_queue[self.operator_locations.index(_)].append(
                self.init_queues[self.operator_locations.index(_)][0])

            # add to items_on_conv
            self.items_on_conv.append(
                [_, self.init_queues[self.operator_locations.index(_)][0], 45 / self.max_time_in_system])

        # remove from init
        self.init_queues = [item[1:] for item in self.init_queues]

    #### Generate the visual conveyor ##########################################################################################################

    def generate_env(self, no_of_gtp, no_of_output):
        """returns empty env, with some variables about the size, lanes etc."""
        empty = np.zeros(
            (15, 4 * no_of_gtp + 4 + 3 * no_of_output + 2, 3))  # height = 15, width dependent on amount of stations
        for i in range(2, empty.shape[1] - 2):
            empty[2][i] = (255, 255, 255)  # toplane = 2
            empty[7][i] = (255, 255, 255)  # bottom lane = 7
        for i in range(2, 8):
            empty[i][1] = (255, 255, 255)  # left lane
            empty[i][empty.shape[1] - 2] = (255, 255, 255)  # right lane

        for i in range(8, empty.shape[0]):
            for j in range(4, no_of_gtp * 4 + 1, 4):  # GTP lanes
                empty[i][j] = (255, 255, 255)  # Gtp in
                empty[i][j - 1] = (250, 250, 250)  # gtp out
            for j in range(empty.shape[1] - no_of_output * 2 - 1, empty.shape[1] - 2, 2):  # order carrier lanes
                empty[i][j] = (255, 255, 255)
        for i in range(4, no_of_gtp * 4 + 1, 4):  # divert and merge points
            empty[7][i - 1] = (255, 242, 229)  # merge
            empty[7][i] = (255, 242, 229)  # divert

        for i in range(empty.shape[1] - no_of_output * 2 - 1, empty.shape[1] - 2, 2):  # output points
            empty[7][i] = (255, 242, 229)

        return empty

    ###### HELPER FUNCTIONS ###########################################################################################################################

    def update_queues(self, quenr, variable):
        'For a given queue 1-3, add a variable (1,2,3)'
        for i in range(self.amount_of_gtps):
            if quenr == i + 1:
                self.init_queues[i].append(variable)

    def remove_from_queue(self, quenr):
        'For a given queue 1-3, remove the first in the queue'
        for i in range(self.amount_of_gtps):
            if quenr == i + 1:
                self.init_queues[i] = self.init_queues[i][1:]

    def add_to_in_que(self, que_nr, to_add):
        'for a given queue, add item to the queue'
        self.in_queue[que_nr].append(to_add)

    def encode(self, var):
        """encodes categorical variables 0-3 to binary"""
        return (0, 0) if var == 0 else (0, 1) if var == 1 else (1, 0) if var == 2 else (1, 1) if var == 3 else var

    def calc_output_reward(self, sigma2=25, mu_optimal=6):
        """
        For a given state, calculate reward for outputting an item on the conveyor
        Uses a normal distribution arround the optimal amount on conveyor (mu_optimaL)
        other vars used:
        - self.init_queues
        - self.items_on_conv
        - self.max_output_reward                #for the reward given (found in config)
        - self.negative_reward_for_flooding     # for prevention of flooding
        """
        sigma = math.sqrt(sigma2)
        amount_in_demand = len([item for sublist in self.init_queues for item in sublist])
        amount_on_conv = len([item for item in self.items_on_conv if item[0][1] < 8])
        amount_to_do = amount_in_demand - amount_on_conv

        if amount_to_do < mu_optimal:
            logging.debug('amount to do is < {}'.format(mu_optimal))
            mu = amount_to_do
            if mu <= 0:
                returned_reward = self.negative_reward_for_flooding  # in this case, you dont want a new output, so punish
            else:
                returned_reward = ((1 / (sigma * math.sqrt(2 * math.pi))) * math.e ** (
                            (-1 / 2) * ((amount_on_conv - mu) / sigma) ** 2)
                                   - (1 / (sigma * math.sqrt(2 * math.pi))) * math.e ** (
                                               (-1 / 2) * ((0 - mu) / sigma) ** 2)) * (1 / ((1 / (sigma * math.sqrt(2 * math.pi))) * math.e ** ((-1 / 2) * ((mu - mu) / sigma) ** 2) - (
                            1 / (sigma * math.sqrt(2 * math.pi))) * math.e ** ((-1 / 2) * ((0 - mu) / sigma) ** 2))) * self.max_output_reward
            logging.debug('mu == {}, returned reward {}'.format(mu, returned_reward))

        else:
            mu = mu_optimal
            returned_reward = ((1 / (sigma * math.sqrt(2 * math.pi))) * math.e ** (
                    (-1 / 2) * ((amount_on_conv - mu) / sigma) ** 2)
                               - (1 / (sigma * math.sqrt(2 * math.pi))) * math.e ** (
                                       (-1 / 2) * ((0 - mu) / sigma) ** 2)) * (1 / (
                        (1 / (sigma * math.sqrt(2 * math.pi))) * math.e ** ((-1 / 2) * ((mu - mu) / sigma) ** 2) - (
                        1 / (sigma * math.sqrt(2 * math.pi))) * math.e ** (
                                    (-1 / 2) * ((0 - mu) / sigma) ** 2))) * self.max_output_reward
            logging.debug('mu == {}, returned reward {}'.format(mu, returned_reward))
        return returned_reward

#####################################################################################
## Make Observation
#
    def make_observation(self):
        '''Builds the observation from the available variables'''

        ### For the obeservation of the conveyor ########################################################################
        # Get all items on lower part of conveyor
        lowerpart = [[item[0][0], item[1]] for item in self.items_on_conv if item[0][1] == 7]

        all_types = []
        lead_observed = []
        # all type 1 for GTP3
        for location in self.diverter_locations:
            amount_of_type1 = len([item for item in lowerpart if item[1] == 1 and item[0] >= location[0]])
            amount_of_type2 = len([item for item in lowerpart if item[1] == 2 and item[0] >= location[0]])
            amount_of_type3 = len([item for item in lowerpart if item[1] == 3 and item[0] >= location[0]])
            all_types.append((amount_of_type1 / (self.amount_of_gtps * 4 + 11)))
            all_types.append((amount_of_type2 / (self.amount_of_gtps * 4 + 11)))
            all_types.append((amount_of_type3 / (self.amount_of_gtps * 4 + 11)))

            bottom_row = [item for item in self.items_on_conv if item[0][1] == 7]                                          #maybe redundant with lowerpart variable
            choice_row = [item for item in bottom_row if item[0][0] >= location[0]]
            try:
                demand = self.init_queues[self.diverter_locations.index(location)][0]
            except:
                demand = 0

            if demand == 0:
                lead_observed.append(1.0)                                                                               #Seems somewhat logical to use 0 as well, experiment
            else:
                extra = 5 if demand == 1 else 3 if demand == 2 else 1
                if demand in [item[1] for item in choice_row]:
                    dist = [item[0][0] for item in choice_row if
                            item[1] == self.init_queues[self.diverter_locations.index(location)][0]][0] - location[0]
                    total_dist = self.output_locations[demand - 1][0] - location[0] + extra
                    dist_normalized = dist / total_dist
                    lead_observed.append(dist_normalized)
                else:
                    dist = self.output_locations[demand - 1][0] - location[0] + extra
                    total_dist = self.output_locations[demand - 1][0] - location[0] + extra
                    dist_normalized = dist / total_dist
                    lead_observed.append(dist_normalized)
        tot_conv = all_types + lead_observed
        on_conv = np.array(tot_conv)

        ### For the observation of the items in queue ##################################################################
                    #length of each queue (how full)            #some indicator of how long it takes to process this full queue (consider 1- x)
        in_queue = [len(item)* 1/7 for item in self.in_queue] + [sum(item)* 1/(3*7) for item in self.in_queue]

        ### For the observation of the demand of the GtP Queue #########################################################
        #make the init list
        init = []
        for item in self.init_queues:
            init1 = item[:self.in_que_observed]
            init.append(init1 + [0] * (self.in_que_observed - len(init1)))
        init = list(np.array(init).flatten())
        # binary encoding of the categorical variables
        init = np.array([self.encode(item) for item in init]).flatten()
        logging.debug('init lenght = {}'.format(len(init)))

        ### Combine All to one array ###################################################################################

        obs = np.append(np.array(init).flatten(), in_queue)         #combine GTP queue with the conveyor memory
        obs = np.append(on_conv, obs)                                  #add the information about what is in queue
        logging.debug('size of observation is: {}'.format(len(obs)))
        return obs

    def make_observation1(self):
        """
        A Second function to make an observation of the system. Observing:
        1. what is on the conveyor (in types)
        2. what is in queue at the moment
            (3. what is in demand still)
        """
        #counts of the amounts of each type on the conveyor
        on_conv = [item[1] for item in self.items_on_conv if item[0][1] < 8]
        amount_t1 = len([item for item in on_conv if item == 1])
        amount_t2 = len([item for item in on_conv if item == 2])
        amount_t3 = len([item for item in on_conv if item == 3])
        total_counts = [amount_t1, amount_t2, amount_t3]

        #amounts of each type in queue
        queue_amounts = []
        for queue in env.in_queue:
            for i in range(3):
                queue_amounts.append(len([box for box in queue if box == i]))

        return np.array(total_counts + queue_amounts)

        #

    ########################################################################################################################################################
    ## RESET FUNCTION
    #
    def reset(self):
        """reset all the variables to zero, empty queues
        must return the current state of the environment"""
        self.run_count += 1  # for logging
        self.episode += 1  # for logging
        #('Ep: {:5}, steps: {:3}, R: {:3.3f}'.format(self.episode, self.steps, self.reward), end='\r')
        self.D_states = {}
        for i in range(1, len(self.diverter_locations) + 1):
            self.D_states[i] = False

        # reset output points
        self.O_states = {}
        for i in range(1, len(self.output_locations) + 1):
            self.O_states[i] = 0

        # reset transition points: 0=no transition, 1=transition
        self.T_states = {}
        for i in range(1, len(self.operator_locations) + 1):
            self.T_states[i] = False

        # reset merge points
        self.M_states = {}
        for i in range(1, len(self.merge_locations) + 1):
            self.M_states[i] = False

        ####### FOR SIMULATION ONLY
        # reset the time to initial values
        self.W_times = {}
        for i in range(1, len(self.operator_locations) + 1):
            self.W_times[i] = self.process_time_at_GTP + 8 * self.amount_of_gtps - 5
        logging.debug("Process times at operator are: {}".format(self.W_times))
        ####### FOR SIMULATION ONLY

        # empty amount of items on conv.
        self.items_on_conv = []
        self.reward = 0
        self.total_travel = 0
        self.steps = 0
        self.terminate = False
        self.step_reward_p = 0
        self.step_reward_n = 0

        # reset tracers
        self.amount_of_items_on_conv = 0
        self.amount_of_items_in_sys = 0
        self.remaining_demand = 0
        self.amount_of_orders_processed = 0
        self.positive_reward = 0
        self.negative_reward = 0
        self.cycle_count = 0
        self.previous_action = 0
        self.same_action_count = 0

        # reset the queues, initialize with a new random set of order sequences
        self.queues = [random.choices(np.arange(1, self.amount_of_outputs + 1),
                                      [self.percentage_small_carriers, self.percentage_medium_carriers,
                                       self.percentage_large_carriers], k=self.gtp_buffer_size) for item in
                       range(self.amount_of_gtps)]  # generate random queues
        self.init_queues = copy(self.queues)
        self.demand_queues = copy(self.queues)
        self.in_queue = [[] for item in range(len(self.queues))]
        self.empty_env = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)
        self.carrier_type_map = np.zeros((self.empty_env.shape[0], self.empty_env.shape[1], 1))

        # do warm start
        self.warm_start()

        return self.make_observation()

    ########################################################################################################################################################
    ## PROCESSING OF ORDER CARRIERS AT GTP
    #
    def process_at_GTP(self):
        # for each step; check if it needed to process an order carrier at GTP
        O_locs = copy(self.operator_locations)
        # check for each operator location (also transition point) if:
        for Transition_point in O_locs:  # For all operator locations, check:

            try:
                # if the order carrier is not equal to the demanded type; set condition to transfer to true
                if self.demand_queues[O_locs.index(Transition_point)][0] != \
                        self.in_queue[O_locs.index(Transition_point)][0]:
                    self.condition_to_transfer = True
                # if the order carrier is equal to the demanded type; set condition to process to true
                elif self.demand_queues[O_locs.index(Transition_point)][0] == \
                        self.in_queue[O_locs.index(Transition_point)][0]:
                    self.condition_to_process = True
            except:
                # else: set to false
                self.condition_to_transfer = False
                self.condition_to_process = False

            # if processingtime at a gtp station == 0 , an order carrier is processed (removed)
            if self.W_times[O_locs.index(Transition_point) + 1] == 0:  # if the waiting time is 0:
                logging.debug('Waiting time at GTP {} is 0, check done on correctness:'.format(
                    O_locs.index(Transition_point) + 1))
                if random.random() < self.exception_occurence:  # if the random occurence is below exception occurence (set in config) do:
                    # remove an order carrier (broken)
                    logging.debug('With a change percentage an order carrier is removed')
                    logging.info('transition point is: {}'.format(Transition_point))
                    # self.update_queues(O_locs.index(Transition_point)+1, [item[1] for item in self.items_on_conv if item[0] == Transition_point][0])
                    # set waiting time to 1 (next step you want to process again)
                    self.W_times[O_locs.index(Transition_point) + 1] = 1
                    # self.O_states[[item[1] for item in self.items_on_conv if item[0] == Transition_point][0]] +=1
                    # remove item from conveyor (e.g. because it is broken)
                    self.items_on_conv = [item for item in self.items_on_conv if item[0] != Transition_point]

                elif self.condition_to_transfer:
                    # move order carrier back onto system via transfer - merge
                    for item in self.items_on_conv:
                        if item[0] == Transition_point:
                            item[0][0] -= 1
                    self.W_times[O_locs.index(
                        Transition_point) + 1] = 1  # set waiting time to 1; you want to check next item next step

                    self.update_queues(O_locs.index(Transition_point) + 1,
                                       self.in_queue[O_locs.index(Transition_point)][
                                           0])  # queues are updated to match situation
                elif self.condition_to_process:
                    # Process an order at GTP successfully
                    logging.debug('Demand queues : {}'.format(self.demand_queues))
                    logging.debug('In queue : {}'.format(self.in_queue))
                    logging.debug('items on conveyor : {}'.format(self.items_on_conv))
                    logging.debug('right order carrier is at GTP (location: {}'.format(Transition_point))
                    logging.debug('conveyor memory before processing: {}'.format(self.items_on_conv))
                    self.items_on_conv = [item for item in self.items_on_conv if item[0] != Transition_point]
                    self.reward += 10 + 10 + (O_locs.index(
                        Transition_point) + 1 * 4) * self.travelpath_to_gtp_reward  # a reward is given for processing an order (can be turned off in config if travelpath_to_gtp_reward = 0)
                    self.step_reward_p += 10 + 10 + (O_locs.index(Transition_point) + 1 * 4)
                    self.positive_reward += 10 + 10 + (
                                O_locs.index(Transition_point) + 1 * 4) * self.travelpath_to_gtp_reward

                    self.amount_of_orders_processed += 1  # for logging
                    logging.debug('order at GTP {} processed'.format(O_locs.index(Transition_point) + 1))
                    logging.debug('conveyor memory after processing: {}'.format(self.items_on_conv))

                    # when processed, remove order carrier from demand queue
                    try:
                        # remove from demand queue
                        self.demand_queues[O_locs.index(Transition_point)] = self.demand_queues[
                                                                                 O_locs.index(Transition_point)][1:]
                    except:
                        logging.info("Except: Demand queue for this lane is allready empty")

                    # set new timestep for the next order
                    try:
                        next_type = [item[1] for item in self.items_on_conv if
                                     item[0] == [Transition_point[0], Transition_point[1] - 1]][0]

                    except:
                        next_type = 99
                    # set new waiting time; based on size of order carrier that is currently processed
                    self.W_times[O_locs.index(
                        Transition_point) + 1] = self.process_time_at_GTP if next_type == 1 else self.process_time_at_GTP + 30 if next_type == 2 else self.process_time_at_GTP + 60 if next_type == 3 else self.process_time_at_GTP + 60 if next_type == 4 else self.process_time_at_GTP + 60
                    logging.debug('new timestep set at GTP {} : {}'.format(O_locs.index(Transition_point) + 1,
                                                                           self.W_times[
                                                                               O_locs.index(Transition_point) + 1]))
                else:
                    logging.debug('Else statement activated')

                # remove from in_queue when W_times is 0
                try:
                    # remove item from the In_que list
                    self.in_queue[O_locs.index(Transition_point)] = self.in_queue[O_locs.index(Transition_point)][1:]
                    logging.debug('item removed from in-que')
                except:
                    logging.debug("Except: queue was already empty!")
            # this should not happen, but a condition to fix in case it does occur:
            elif self.W_times[O_locs.index(Transition_point) + 1] < 0:
                self.W_times[O_locs_locations.index(Transition_point) + 1] = 0
                logging.debug("Waiting time was below 0, reset to 0")
            else:
                self.W_times[O_locs.index(Transition_point) + 1] -= 1  # decrease timestep with 1
                logging.debug('waiting time decreased with 1 time instance')
                logging.debug('waiting time at GTP{} is {}'.format(O_locs.index(Transition_point) + 1,
                                                                   self.W_times[O_locs.index(Transition_point) + 1]))

    ########################################################################################################################################################
    ## STEP FUNCTION
    #
    def step_env(self):
        """
        Function that initiates 1 time advancement in the system. All items move one step. also possible action are done based on conditions.
        Is ran for each action (each timestep)
        """

        ####make carrier type map (observe the current state again)
        self.carrier_type_map = np.zeros((self.empty_env.shape[0], self.empty_env.shape[1], 1)).astype(float)
        for item in self.items_on_conv:
            self.carrier_type_map[item[0][1]][item[0][0]] = item[1]
            item[2] += 1 / self.max_time_in_system  # increase the time in the system

        # give a negative reward for each step, for all the items that are taking are taking a loop
        # self.reward += len([item for item in self.items_on_conv if item[0][1] < 7]) * self.negative_reward_per_step       #tag:punishment (currently not used)

        #### Process the orders at GTP > For simulation: do incidental transfer of order carrier
        self.process_at_GTP()

        ####toggle diverters if needed
        # toggle All D_states if needed:
        d_locs = copy(self.diverter_locations)
        carrier_map = copy(self.carrier_type_map)
        for loc2 in d_locs:
            try:
                # Condition 1 = if the carrier type at any of the diverter locations is EQUAL TO the next-up requested carrier type at GTP request lane of this specific diverter location
                condition_1 = carrier_map[loc2[1]][loc2[0]] == self.init_queues[d_locs.index(loc2)][0]
                if condition_1 == True:
                    self.D_condition_1[d_locs.index(loc2) + 1] = True
                else:
                    self.D_condition_1[d_locs.index(loc2) + 1] = False
                # condition 2 = if the lenght of the in_queue is <= smallest queue that also demands order carrier of the same type
                # condition_2 = len(self.in_queue[d_locs.index(loc2)])-1 <= min(map(len, self.in_queue))
                if carrier_map[loc2[1]][loc2[0]] != 0:
                    logging.debug('Left condition at lane {} = {}'.format(d_locs.index(loc2) + 1,
                                                                          len(self.in_queue[d_locs.index(loc2)])))
                    logging.debug('Right condition at lane {} = {}'.format(d_locs.index(loc2) + 1, min(map(len, [
                        self.in_queue[self.init_queues.index(i)] for i in
                        [item for item in [item for item in self.init_queues if item != []] if
                         item[0] == self.init_queues[d_locs.index(loc2)][0]]]))))
                    condition_2 = len(self.in_queue[d_locs.index(loc2)]) <= min(map(len, [
                        self.in_queue[self.init_queues.index(i)] for i in
                        [item for item in [item for item in self.init_queues if item != []] if
                         item[0] == self.init_queues[d_locs.index(loc2)][0]]]))
                else:
                    condition_2 = False
                logging.debug('Condition 2 == {}'.format(condition_2))
                if condition_2 == True:
                    self.D_condition_2[d_locs.index(loc2) + 1] = True
                else:
                    self.D_condition_2[d_locs.index(loc2) + 1] = False
                # check if the next space is empty (queue is not full)
                condition_3 = carrier_map[loc2[1] + 1][loc2[0]] == 0
                logging.debug(carrier_map[loc2[1] + 1][loc2[0]] == 0)

                # if all conditions apply, an order carrier is set ready to move into queue
                if condition_1 and condition_2 and condition_3:
                    self.D_states[d_locs.index(loc2) + 1] = True
                    logging.debug("set diverter state for diverter {} to TRUE".format(d_locs.index(loc2) + 1))
                    self.remove_from_queue(d_locs.index(loc2) + 1)
                    logging.debug("request removed from demand queue")
                    self.add_to_in_que(d_locs.index(loc2), int(carrier_map[loc2[1]][loc2[0]]))
                    logging.debug("Order carrier added to GTP queue")

                else:
                    self.D_states[d_locs.index(loc2) + 1] = False
                    logging.debug("Divert-set requirement not met at cord {}".format(loc2))
            except IndexError:
                logging.debug('Index error: queues are empty!')
                self.D_states[d_locs.index(loc2) + 1] = False
            except:
                logging.warning('Another error occurred; this should not happen! Investigate the cause!')
                self.D_states[d_locs.index(loc2) + 1] = False

        ####Divert when at diverter, and diverter is set to true
        ####Do step for all items on conv
        for item in self.items_on_conv:
            # check diverters; is any of them True and is there an order carrier? move into lane.
            if item[0] in self.diverter_locations:
                try:
                    if self.D_states[self.diverter_locations.index(item[0]) + 1] == True and \
                            self.carrier_type_map[item[0][1] + 1][item[0][0]] == 0:
                        item[0][1] += 1
                        self.reward += self.positive_reward_for_divert  # if order carrier is correctly delivered, reward is given
                        self.step_reward_p += self.positive_reward_for_divert
                        logging.debug(
                            "moved order carrier into GTP lane {}".format(self.diverter_locations.index(loc1) + 1))
                    else:
                        item[0][0] -= 1
                except:
                    logging.debug("Item of size {} not moved into lane: Divert value not set to true".format(item[1]))

                    # otherwise; all items set a step in their moving direction
            elif item[0][1] == 7 and item[0][
                0] > 1:  # and self.carrier_type_map[item[0][1]][item[0][0]-1] ==0: #if on the lower line, and not reached left corner:
                item[0][0] -= 1  # move left
                logging.debug('item {} moved left'.format(item[0]))
            elif item[0][0] == 1 and item[0][
                1] > 2:  # and self.carrier_type_map[item[0][1]-1][item[0][0]] ==0: #if on left lane, and not reached top left corner:
                item[0][1] -= 1
                logging.debug('item {} moved up'.format(item[0]))  # move up
            elif item[0][1] == 2 and item[0][0] < self.empty_env.shape[
                1] - 2:  # and self.carrier_type_map[item[0][1]][item[0][0]+1] ==0: #if on the top lane, and not reached right top corner:
                item[0][0] += 1  # Move right
                logging.debug('item {} moved right'.format(item[0]))
            elif item[0][0] == self.empty_env.shape[1] - 2 and item[0][
                1] < 7:  # and self.carrier_type_map[item[0][1]+1][item[0][0]] ==0: #if on right lane, and not reached right down corner:
                item[0][1] += 1
                logging.debug('item {} moved down'.format(item[0]))
            elif item[0][1] > 7 and item[0][0] in [lane[0] for lane in self.diverter_locations] and item[0][1] < \
                    self.empty_env.shape[0] - 1 and item[0][0] < self.amount_of_gtps * 4 + 3 and \
                    self.carrier_type_map[item[0][1] + 1][item[0][0]] == 0:  # move down into lane
                item[0][1] += 1
                logging.debug('item {} moved into lane'.format(item[0]))
            elif item[0][1] > 7 and item[0][0] in [lane[0] for lane in self.merge_locations] and item[0][
                0] < self.amount_of_gtps * 4 + 3 and self.carrier_type_map[item[0][1] - 1][item[0][0]] == 0 and \
                    self.carrier_type_map[item[0][1] - 1][item[0][0] + 1] == 0:  # move up into merge lane
                item[0][1] -= 1
            elif item[0][1] > 7 and item[0][0] > self.amount_of_gtps * 4 + 3 and self.carrier_type_map[item[0][1] - 1][
                item[0][0]] == 0:  # move up if on output lane
                item[0][1] -= 1
                logging.debug('item {} moved onto conveyor'.format(item[0]))

        ####try to add new item from output when On!=0
        for cord2 in self.output_locations:
            loc = copy(cord2)
            if self.O_states[self.output_locations.index(loc) + 1] != 0 and self.carrier_type_map[cord2[1]][
                cord2[0] + 1] == 0:
                self.items_on_conv.append([loc, self.output_locations.index(loc) + 1, 0])
                self.O_states[self.output_locations.index(loc) + 1] -= 1
                logging.debug("Order carrier outputted at {}".format(loc))
                logging.debug("Items on conveyor: {}".format(self.items_on_conv))
            elif self.O_states[self.output_locations.index(loc) + 1] != 0 and self.carrier_type_map[cord2[1]][
                cord2[0] + 1] != 0:
                logging.debug('Not able to output on output {} .'.format(loc))
                # because not able to output, set the output state back to 0
                self.O_states[self.output_locations.index(loc) + 1] = 0
                # give a negative reward for trying this action
                self.reward += self.negative_reward_for_invalid
                self.step_reward_n += self.negative_reward_for_invalid

    def step(self, action):
        """
        Generic step function; takes a step bij calling step_env()
        observation is returned
        Reward is calculated
        Tracers are logged
        Termination case is determined

        returns state, reward, terminate, {}
        """
        # print("", end='\r')
        if action == self.previous_action:
            self.same_action_count += 1
        elif action != self.previous_action:
            self.same_action_count = 0
        self.previous_action = action
        # print('Action: {} Step {}'.format(action, self.steps), end='\r')
        self.step_reward_p = 0
        self.step_reward_n = 0
        logging.debug("Executed action: {}".format(action))
        logging.debug("Demand queue: {}".format(self.demand_queues))
        self.steps += 1
        if action == 0:
            self.step_env()
            logging.debug("- - action 0 executed")
            logging.debug("Divert locations :{}".format(self.diverter_locations))
            logging.debug('states of Divert points = {}'.format(self.D_states))

        elif action == 1:
            if len([item[1] for item in self.items_on_conv if item[0][1] < 8 and item[1] == 1]) == len(
                    [item for sublist in self.init_queues for item in sublist if item == 1]):
                self.reward += self.negative_reward_for_flooding
                self.step_reward_n += self.negative_reward_for_flooding
            else:
                self.O_states[1] += 1
                self.step_env()
                self.reward += self.calc_output_reward()

            logging.debug("- - action 1 executed")
        elif action == 2:
            if len([item[1] for item in self.items_on_conv if item[0][1] < 8 and item[1] == 2]) == len(
                    [item for sublist in self.init_queues for item in sublist if item == 2]):
                self.reward += self.negative_reward_for_flooding
                self.step_reward_n += self.negative_reward_for_flooding
            else:
                self.O_states[2] += 1
                self.step_env()
                self.reward += self.calc_output_reward()

            logging.debug("- - action 2 executed")
        elif action == 3:
            if len([item[1] for item in self.items_on_conv if item[0][1] < 8 and item[1] == 3]) == len(
                    [item for sublist in self.init_queues for item in sublist if item == 3]):
                self.reward += self.negative_reward_for_flooding
                self.step_reward_n += self.negative_reward_for_flooding
            else:
                self.O_states[3] += 1
                self.step_env()
                self.reward += self.calc_output_reward()
            logging.debug("- - action 3 executed")
        else:
            self.reward -= 20  # tag:punishment
            self.step_reward_n -= 20
            print('Terminate because of invalid action handling!')
            self.terminate = True

        logging.debug("states of O: {}".format(self.O_states))
        logging.debug("init queues :{}".format(self.init_queues))
        logging.debug('amount of items in init queues: {}'.format(sum([len(item) for item in self.init_queues])))
        logging.debug("conveyor memory : {}".format(self.items_on_conv))
        logging.debug('Amount of items on conv: {}'.format(len(self.items_on_conv)))
        logging.debug('Demand queue : {}'.format(self.demand_queues))
        logging.debug(
            'amount of demand still needing processing: {}'.format(sum([len(item) for item in self.demand_queues])))
        logging.debug('In queue : {}'.format(self.in_queue))
        logging.debug('In queue items : {} '.format(sum([len(item) for item in self.in_queue])))
        logging.debug(
            'In queue according to convmap : {}'.format(len([item for item in self.items_on_conv if item[0][1] > 7])))
        logging.debug('')
        logging.debug(
            '--------------------------------------------------------------------------------------------------------------------')

        next_state = self.make_observation()

        ### calculate conditional reward ##############################################################################
        if len([item for item in self.items_on_conv if
                item[0] == [1, 7]]) == 1:  # in case that negative reward is calculated with cycles
            self.reward += self.negative_reward_for_cycle  # punish if order carriers take a cycle #tag:punishment
            self.step_reward_n += self.negative_reward_for_cycle
            self.cycle_count += 1

        # if len([item for item in self.items_on_conv if item[0][1] < 8]) > len([item for sublist in self.init_queues for item in sublist]):
        #     self.reward += self.negative_reward_for_flooding                                                            #tag:punishment
        ## FOR FLOODING
        for i in range(1, self.amount_of_outputs + 1):
            if len([item[1] for item in self.items_on_conv if item[0][1] < 8 and item[1] == i]) > len(
                    [item for sublist in self.init_queues for item in sublist if item == i]):
                logging.debug('Too many items of type {} on conv! - punished!'.format(i))
                self.reward += self.negative_reward_for_flooding
                self.step_reward_n += self.negative_reward_for_flooding
            else:
                logging.debug('Not too many items of type {} on conv!'.format(i))

        # punishment for having empty queues
        for queue in self.in_queue:
            if len(queue) == 0:
                self.reward += self.negative_reward_for_empty_queue  # tag:punishment
                self.step_reward_n += self.negative_reward_for_empty_queue

        ### Define some tracers     ##################################################################################
        self.amount_of_items_on_conv = len([item for item in self.items_on_conv if item[0][1] < 8])
        self.amount_of_items_in_sys = len(self.items_on_conv)
        self.remaining_demand = len([item for sublist in self.demand_queues for item in sublist])

        ### Determine Termination cases ###############################################################################
        # try:
        #     if max([item[2] for item in self.items_on_conv]) >= 1:
        #         self.terminate = True
        # except:
        #     self.terminate = False

        # terminate if the demand queues are empty (means all is processed)
        if self.termination_condition == 1:
            if self.demand_queues == [[] * i for i in range(self.amount_of_gtps)]:
                self.terminate = True
        elif self.termination_condition == 2:
            if self.init_queues == [[] * i for i in range(self.amount_of_gtps)]:
                self.terminate = True

        if self.same_action_count > 1000:                  #when more then 10.000 same steps done, terminate
            self.terminate = True

        ### Summarized return for step function ####################################################################
        reward = self.reward
        terminate = self.terminate
        logging.debug('Reward is: {}'.format(self.reward))
        return next_state, reward, terminate, {}

    ################## RENDER FUNCTIONS ################################################################################################
    def render_plt(self):
        """Simple render function, uses matplotlib to render the image in jupyter notebook + some additional information on the transition points"""
        # print('items on conveyor:')
        # print(self.items_on_conv)
        # print('states of Divert points = {}'.format(self.D_states))
        # print('states of Output points = {}'.format(self.O_states))
        # for queue in self.init_queues:
        # print('Queue GTP{}: {}'.format(self.init_queues.index(queue), queue))

        self.image = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)

        for item in self.items_on_conv:
            self.image[item[0][1]][item[0][0]] = np.asarray([self.pallette[0] if item[1] == 1 else self.pallette[1] if
            item[1] == 2 else self.pallette[2] if item[1] == 3 else self.pallette[3]])
        self.image = self.image / 255.0
        plt.imshow(np.asarray(self.image))
        plt.show()

    def render(self, mode='human'):
        """render with opencv, for eyeballing while testing"""
        resize_factor = 36
        box_diameter = 30
        self.image = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)
        im = Image.fromarray(np.uint8(self.image))
        img = im.resize((self.image.shape[1] * resize_factor, self.image.shape[0] * resize_factor),
                        resample=Image.BOX)  # BOX for no anti-aliasing)
        draw = ImageDraw.Draw(img)

        for i in range(7):
            for item in copy(self.output_locations):
                x0 = item[0] * resize_factor + 3
                y0 = item[1] * resize_factor + 40 + 3 + i * 35
                box_size = 20 if item == self.output_locations[0] else 25 if item == self.output_locations[
                    1] else 30 if item == self.output_locations[2] else 30
                x1 = x0 + box_size
                y1 = y0 + box_size
                color = self.pallette[0] if item == self.output_locations[0] else self.pallette[1] if item == \
                                                                                                      self.output_locations[
                                                                                                          1] else \
                self.pallette[2] if item == self.output_locations[2] else self.pallette[2]
                draw.rectangle([x0, y0, x1, y1], fill=tuple(color), outline='black')

        # Draw the order carriers
        for item in self.items_on_conv:
            size = box_diameter - 10 if item[1] == 1 else box_diameter - 5 if item[1] == 2 else box_diameter if item[
                                                                                                                    1] == 3 else box_diameter
            x0 = item[0][0] * resize_factor + 3
            x1 = x0 + size
            y0 = item[0][1] * resize_factor + 3
            y1 = y0 + size
            color = self.pallette[0] if item[1] == 1 else self.pallette[1] if item[1] == 2 else self.pallette[2] if \
            item[1] == 3 else self.pallette[3]
            draw.rectangle([x0, y0, x1, y1], fill=tuple(color), outline='black')

        # Draw demands
        for item in copy(self.diverter_locations):
            x0 = item[0] * resize_factor + 40
            y0 = item[1] * resize_factor + 40
            x1 = x0 + 30
            y1 = y0 + 30

            try:
                next_up = self.init_queues[self.diverter_locations.index(item)][0]
            except:
                next_up = '-'
            color = self.pallette[0] if next_up == 1 else self.pallette[1] if next_up == 2 else self.pallette[
                2] if next_up == 3 else (225, 225, 225)
            draw.ellipse([x0, y0, x1, y1], fill=tuple(color), outline=None)
            draw.text((x0 + 10, y0 + 5), '{}'.format(next_up), fill='black',
                      font=ImageFont.truetype(font='arial', size=20, index=0, encoding='unic', layout_engine=None))
            draw.text((x0, y1 + 5), 'Demand \n Queue',
                      font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))

            # draw demand conditions
            x2, y2 = item[0] * resize_factor, item[1] * resize_factor - 12
            x3, y3 = x2 + 10, y2 + 10
            x4, y4 = x3 + 5, y2
            x5, y5 = x4 + 10, y4 + 10
            color1 = 'green' if self.D_condition_1[self.diverter_locations.index(item) + 1] == True else 'red'
            color2 = 'green' if self.D_condition_2[self.diverter_locations.index(item) + 1] == True else 'red'
            draw.ellipse([x2, y2, x3, y3], fill=color1, outline=None)
            draw.ellipse([x4, y4, x5, y5], fill=color2, outline=None)

            # init queues on top
            x6, y6 = item[0] * resize_factor - 30, item[1] * resize_factor - 30
            draw.text((x6 + 10, y6 + 5), '{}'.format(self.init_queues[self.diverter_locations.index(item)][:5]),
                      fill='white',
                      font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))

            # in_queue
            x7, y7 = x0, y0 + 95
            draw.text((x7, y7), 'In queue', fill='white',
                      font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))
            draw.text((x7, y7 + 15), '{}'.format(self.in_queue[self.diverter_locations.index(item)][:5]), fill='white',
                      font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))

        ##### TURN OFF FOR FASTER RENDERING #############################################################################################
        # values of the O_states
        for item in copy(self.output_locations):
            x0 = item[0] * resize_factor + 40
            y0 = item[1] * resize_factor + 40
            draw.text((x0, y0), '{}'.format(self.O_states[self.output_locations.index(item) + 1]), fill='white',
                      font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))

        # draw reward
        x0, y0 = self.diverter_locations[0][0] * resize_factor + 130, self.diverter_locations[0][
            1] * resize_factor + 150
        y1 = y0 + 25
        y2 = y1 + 25
        draw.text((x0, y0), ' Total Reward: {}'.format(self.reward), fill='white',
                  font=ImageFont.truetype(font='arial', size=15, index=0, encoding='unic', layout_engine=None))
        draw.text((x0, y1), ' Positive Reward: {}'.format(self.step_reward_p), fill='green',
                  font=ImageFont.truetype(font='arial', size=15, index=0, encoding='unic', layout_engine=None))
        draw.text((x0, y2), ' Negative Reward: {}'.format(self.step_reward_n), fill='red',
                  font=ImageFont.truetype(font='arial', size=15, index=0, encoding='unic', layout_engine=None))
        ###################################################################################################################################

        # Draw GTP demands
        for item in copy(self.operator_locations):
            x0 = item[0] * resize_factor + 40
            y0 = item[1] * resize_factor
            x1 = x0 + 30
            y1 = y0 + 30

            try:
                next_up = self.demand_queues[self.operator_locations.index(item)][0]
            except:
                next_up = '-'
            color = self.pallette[0] if next_up == 1 else self.pallette[1] if next_up == 2 else self.pallette[
                2] if next_up == 3 else (225, 225, 225)
            draw.ellipse([x0, y0, x1, y1], fill=tuple(color), outline=None)
            draw.text((x0 + 10, y0 + 5), '{}'.format(next_up), fill='black',
                      font=ImageFont.truetype(font='arial', size=20, index=0, encoding='unic', layout_engine=None))
            draw.text((x0, y0 - 45), 'Demand \n at GtP',
                      font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))

            # demand queues
            draw.text((x0, y0 - 15), '{}'.format(self.demand_queues[self.operator_locations.index(item)][:5]),
                      font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))

        # resize with PIL
        # img = img.resize((1200,480), resample=Image.BOX)
        cv2.imshow(self.window_name, cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    def create_window(self):
        # used for visual training
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
                        '   Episode {:2}, Step {:3}, Reward: {:.2f}, State: {}, Action: {:2}'.format(episode, step,
                                                                                                     reward,
                                                                                                     state[0],
                                                                                                     action[0]),
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

if __name__ == "__main__":
    print('check')