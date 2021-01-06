########################################################################################
# Version: 1.0
#
#  1x1 version
#
#
########################################################################################

import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import copy
import seaborn as sns
import random
import logging
import gym
from PIL import Image, ImageDraw, ImageFont
import math

# CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements
logging.basicConfig(level=logging.INFO)


class NewConveyor10(gym.Env):

    ######## INITIALIZATION OF VARIABLES ###############################################################################
    def __init__(self, config):
        """initialize states of the variables, the lists used"""

        # init configs: for explaination of these variables, check: config/NewConveyor1.yml
        self.config = config['environment']
        self.amount_of_gtps = self.config['amount_of_gtps']
        self.amount_of_outputs = self.config['amount_of_outputs']
        self.gtp_buffer_size = self.config['gtp_buffer_size']
        self.percentage_small_carriers = self.config['percentage_small_carriers']
        self.percentage_medium_carriers = self.config['percentage_medium_carriers']
        self.percentage_large_carriers = self.config['percentage_large_carriers']
        self.process_time_at_GTP = self.config['process_time_at_GTP']
        self.exception_occurrence = self.config['exception_occurrence']
        self.termination_condition = self.config['termination_condition']
        self.window_name = 'New Conveyor render'
        self.max_cycle_count = self.config['max_cycle_count']
        self.in_que_observed = self.config['in_que_observed']

        # init rewards from config
        self.neg_reward_ia = self.config['negative_reward_for_invalid_action']  # - TODO: how to relate to observation
        self.negative_reward_for_cycle = self.config['negative_reward_for_cycle']
        self.shortage_reward = self.config['shortage_reward']
        self.flooding_reward = self.config['flooding_reward']
        self.optimal_pipe = self.config['optimal_pipe']
        self.ranges = [[self.optimal_pipe-3, self.optimal_pipe+1], [self.optimal_pipe-2, self.optimal_pipe+2], [self.optimal_pipe-1, self.optimal_pipe+3]]  # upper and lower bounds for pipelines
        self.wrong_sup_at_goal  = self.config['wrong_sup_at_goal']
        self.positive_reward_for_divert = self.config['positive_reward_for_divert']
        self.reward_empty_queue = self.reward_full_queue = self.config['proper_queue_reward']

        # init variables
        self.episode = 0
        self.reward = 0
        self.terminate = False
        self.items_on_conv = []
        self.empty_env = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)
        self.carrier_type_map = np.zeros((self.empty_env.shape[0], self.empty_env.shape[1], 1))

        #init tracers
        self.steps = 0
        self.previous_action = 0
        self.same_action_count = 0
        self.cycle_count = 0
        self.items_processed = 0

        # init queues
        self.queues = [random.choices(np.arange(1, self.amount_of_outputs + 1),
                                      [0.5, 0.5],
                                      k=self.gtp_buffer_size) for _ in
                       range(self.amount_of_gtps)]  # generate random queues based on distribution defined in config
        logging.info("queues that are initialized: {}".format(self.queues))

        # make required copies
        self.init_queues = copy(self.queues)
        self.demand_queues = copy(self.queues)
        self.in_queue = [[] for _ in range(len(self.queues))]

        self.queue_demand = [item[0] for item in self.init_queues]
        # define locations
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

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(self.amount_of_outputs * self.amount_of_gtps + 1)
        # self.shape = self.amount_of_gtps + 3 * self.amount_of_gtps + (
        #             self.in_que_observed * self.amount_of_gtps * 2) + (2 * self.amount_of_gtps)    # - TODO             #1
        self.shape = self.amount_of_outputs + 3*self.in_que_observed + 1
        self.observation_space = gym.spaces.Box(shape=(self.shape,),
                                                high=1, low=0,
                                                dtype=np.float)

        # colors used in the render
        self.pallette = (np.asarray(sns.color_palette("Reds", self.amount_of_outputs)) * 255).astype(int)

        ####### FOR SIMULATION ONLY
        # a counter to record the processing time at GtP station
        self.W_times = {}
        for i in range(1, len(self.operator_locations) + 1):
            self.W_times[
                i] = self.process_time_at_GTP + 8 * self.amount_of_gtps - 5  # initialize with some time for first run
        logging.debug("Process times at operator are:{}".format(self.W_times))

        self.condition_to_transfer = False
        self.condition_to_process = False

        ## For render
        # D conditions: for the render
        self.D_condition_1 = {}
        for i in range(1, len(self.diverter_locations) + 1):
            self.D_condition_1[i] = False

        self.D_condition_2 = {}
        for i in range(1, len(self.diverter_locations) + 1):
            self.D_condition_2[i] = False
        ####### FOR SIMULATION ONLY

        ######## Do a warm_start
        self.do_warm_start(3)

    def do_warm_start(self, x):
        for _ in range(x):
            self.warm_start()

    def warm_start(self):
        # add items to queues, so queues are not empty when starting with training (empty queue is punished with -1 each timestep)
        for _ in self.operator_locations:
            self.in_queue[self.operator_locations.index(_)].append(
                self.init_queues[self.operator_locations.index(_)][0])

            self.update_init(self.operator_locations.index(_))

            # add to items_on_conv

            if len(self.in_queue[0]) == 1:
                self.items_on_conv.append(
                    [_, self.queue_demand[self.operator_locations.index(_)], self.operator_locations.index(_)+1])
            elif len(self.in_queue[0]) == 2:
                self.items_on_conv.append(
                    [[_[0], _[1]-1], self.queue_demand[self.operator_locations.index(_)], self.operator_locations.index(_) + 1])
            elif len(self.in_queue[0]) == 3:
                self.items_on_conv.append(
                    [[_[0], _[1]-2], self.queue_demand[self.operator_locations.index(_)], self.operator_locations.index(_) + 1])
            elif len(self.in_queue[0]) == 4:
                self.items_on_conv.append(
                    [[_[0], _[1]-3], self.queue_demand[self.operator_locations.index(_)], self.operator_locations.index(_) + 1])
            self.update_queue_demand()

    #### Generate the visual conveyor ##################################################################################

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

    ########### HELPER FUNCTIONS ############################################################################################
    def update_queue_demand(self):
        """Update current demand of queue"""
        new_demand = []
        for item in self.init_queues:
            if item == []:
                new_demand.append(0)
            else:
                new_demand.append(item[0])
        self.queue_demand = new_demand

    def update_init(self, queuenr):
        """Update the queue demand (init_queue)"""
        self.init_queues[queuenr] = self.init_queues[queuenr][1:]

    def update_queues(self, quenr, variable):
        'For a given queue 1-3, add a variable (1,2,3)'
        for i in range(self.amount_of_gtps):
            if quenr == i + 1:
                self.init_queues[i].append(variable)

    def OneHotEncode(self, var):
        """One-hot-encode the types {1,2,3}"""
        if var == 0:
            res = [0, 0, 0]
        elif var == 1:
            res = [1, 0, 0]
        elif var == 2:
            res = [0, 1, 0]
        elif var == 3:
            res = [0, 0, 1]
        return res

    #########################################################################################################################
    ##### Make Observation
    ####
    def make_observation(self):
        '''Builds the observation from the available variables
        1. occupation of the output points - {1,0} - occupied or not e.g. - [0, 1, 1]                       +3 = 3 = amount of output
        2. queue demand - {1,0} - one-hot-encoded - 3 types e.g. [3, 2, 2] > [0, 0, 1, 0, 1, 0, 0, 1, 0]    +9 = 12      > Possibly add more demand
        3. amount of items in pipeline - per pipeline / 25 > [0.16, 0.32, 0.08]                             +3 = 15      > could also say: 1-7 = x/7, more then 7 = 1 | or a share for the amount vs. demand: in_pipe/demand = 1 (o.i.d)
        4. amount of items in queues - per queue /7 > [0.14285714, 0.14285714, 0.        ]                  +3 = 18
        5. amount of itemst that took a cycle / self.max_amount_of_cycles                                   +1 = 19      > could also say; more then x cycles = 1 e.g. 25
        6. Usability of the pipeline (1var)

        '''
        ### 1
        ### For the obeservation of the conveyor ########################################################################
        self.carrier_type_map_obs = np.zeros((self.empty_env.shape[0], self.empty_env.shape[1], 1)).astype(float)

        for item in self.items_on_conv:
            self.carrier_type_map_obs[item[0][1]][item[0][0]] = item[1]
        # cut padding
        type_map_obs = self.carrier_type_map_obs[2:8, 1:-1]  # for the carrier type

        # Only observe bottom lane
        # carrier_type_map_obs = type_map_obs[-1]  # top and bottom lane for the carrier type                             # 25
        # OR
        # carrier_type_map_obs = type_map_obs[-1, -6:]  ## returns: array([[3.],[0.],[3.],[0.],[3.],[0.]])              # 6
        # OR
        carrier_type_map_obs = type_map_obs[-1, -2*self.amount_of_outputs:][
                               ::2]  ## returns: array([[3.],[3.],[3.]])                         # 3
        carrier_type_map_obs = np.array([1 if item != 0 else 0 for item in carrier_type_map_obs])
        logging.debug(carrier_type_map_obs)
        # 2 - queue demand: maybe binary?
        # encoded_demand = [self.OneHotEncode(item) for item in self.queue_demand]
        # obs = np.append(carrier_type_map_obs, np.array(encoded_demand).flatten())

        # OR

        init = []
        for item in self.init_queues:
            init1 = item[:self.in_que_observed]
            init.append(init1 + [0] * (self.in_que_observed - len(init1)))
        init = list(np.array(init).flatten())
        encoded_demand = [self.OneHotEncode(item) for item in init]
        obs = np.append(carrier_type_map_obs, np.array(encoded_demand).flatten())

        #3 - in pipeline (amounts) for each queue
        # in_pipeline = []
        # for i in range(1, self.amount_of_gtps + 1):
        #     in_pipeline.append(len([item for item in self.items_on_conv if item[0][1] <8 and item[2] == i]))
        # obs = np.append(obs, np.array(in_pipeline)/36)

        # 4 - length of queues
        in_queues = [len(item) for item in self.in_queue]
        obs = np.append(obs, np.array(in_queues)/7)

        # 5 - cycle count
        # cyclecount = self.to_binary(self.cycle_count)
        # obs = np.append(obs, np.array(cyclecount + [0]))
        # obs = np.append(obs, np.array(self.cycle_count)/self.max_cycle_count)

        # 6 - usability var
        # tot_in_queue = 0
        # tot_on_conv = 0
        # usability_var = 0
        # for queue in self.init_queues:
        #     for i in range(self.amount_of_outputs):
        #         amount_in_queue = len([item for item in self.init_queues[0] if item == i + 1])
        #         tot_in_queue += amount_in_queue
        #         on_conv = len([item[1] for item in self.items_on_conv if
        #                        item[0][1] < 8 and item[1] == i + 1 and item[2] == self.init_queues.index(queue) + 1])
        #         tot_on_conv += on_conv
        #         if amount_in_queue - on_conv >= 0:
        #             indic = 1
        #             usability_var += indic
        #         elif amount_in_queue - on_conv < 0:
        #             indic = amount_in_queue / on_conv
        #             usability_var += indic
        # usability = usability_var / self.amount_of_outputs
        # obs =np.append(obs, usability)
        return obs

    ########################################################################################################################
    ## RESET FUNCTION
    #
    def reset(self):
        """reset all the variables to zero, empty queues
        must return the current state of the environment"""
        self.episode +=1
        print('Ep: {:5}, steps: {:3}, R: {:3.3f}'.format(self.episode, self.steps, self.reward), end='\r')
        self.reward = 0
        self.terminate = False
        self.items_on_conv = []
        #reset tracer
        self.steps = 0
        self.cycle_count = 0
        self.items_processed = 0

        # reset the queues, initialize with a new random set of order sequences
        self.queues = [random.choices(np.arange(1, self.amount_of_outputs + 1),
                                      [0.5, 0.5], k=self.gtp_buffer_size) for item in
                       range(self.amount_of_gtps)]  # generate random queues
        self.init_queues = copy(self.queues)
        self.demand_queues = copy(self.queues)
        self.in_queue = [[] for item in range(len(self.queues))]
        self.queue_demand = [item[0] for item in self.init_queues]
        self.carrier_type_map = np.zeros((self.empty_env.shape[0], self.empty_env.shape[1], 1))

        # D conditions: for the render
        self.D_condition_1 = {}
        for i in range(1, len(self.diverter_locations) + 1):
            self.D_condition_1[i] = False

        self.D_condition_2 = {}
        for i in range(1, len(self.diverter_locations) + 1):
            self.D_condition_2[i] = False

        self.condition_to_transfer = False
        self.condition_to_process = False

        self.do_warm_start(3)

        return self.make_observation()

    ########################################################################################################################
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
                if random.random() < self.exception_occurrence:  # if the random occurence is below exception occurence (set in config) do:
                    # remove an order carrier (broken)
                    logging.debug('With a change percentage an order carrier is removed')
                    logging.debug('transition point is: {}'.format(Transition_point))
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
                    self.items_processed +=1
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

    ####################################################################################################################
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

        # process orders at GTP
        self.process_at_GTP()

        # do a step for all items on the conveyor
        for item in self.items_on_conv:
            if item[0] in self.diverter_locations:
                if item[1] == self.queue_demand[self.diverter_locations.index(item[0])]:
                    self.D_condition_1[self.diverter_locations.index(item[0])+1] = True
                else:
                    self.D_condition_1[self.diverter_locations.index(item[0]) + 1] = False

                if item[2] == (self.diverter_locations.index(item[0]) + 1):
                    self.D_condition_2[self.diverter_locations.index(item[0]) + 1] = True
                else:
                    self.D_condition_2[self.diverter_locations.index(item[0]) + 1] = False

                # condition 1: if the item at Dloc == the current demand
                condition_1 = item[1] == self.queue_demand[self.diverter_locations.index(item[0])]
                # condition 2: if the goal of the current item == the current gtp queue
                condition_2 = item[2] == (self.diverter_locations.index(item[0]) + 1)
                #condition 3: queue is not full
                condition_3 = [item[0][0], item[0][1]+1] not in [item[0] for item in self.items_on_conv]

                if condition_1 and condition_2 and condition_3:
                    self.update_init(self.diverter_locations.index(item[0]))
                    self.update_queue_demand()
                    self.in_queue[self.diverter_locations.index(item[0])].append(item[1])
                    item[0][1] += 1
                    logging.debug('moved carrier into lane')
                    self.reward += self.positive_reward_for_divert #postive_reward_for_divert
                elif condition_2 and condition_3 and not condition_1:
                    self.reward += self.wrong_sup_at_goal
                    item[0][0] -= 1

                elif condition_1 and condition_2 and not condition_3:
                    self.reward += self.flooding_reward
                    item[0][0] -= 1

                else:
                    item[0][0] -= 1


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

        ## OUTPUT ITEM ON CONVEYOR
        for cord2 in self.output_locations:
            loc = copy(cord2)
            condition1 = self.output_locations.index(
                loc) + 1 == self.next_O  # condition 1: if the output location == Next O
            condition2 = self.carrier_type_map[loc[1]][
                             loc[0] + 1] == 0  # condition 2: if the conveyor is empty to output
            if condition1 and condition2:
                self.items_on_conv.append([loc, self.output_locations.index(loc) + 1, self.next_D])
            if condition1 and condition2 == False:
                self.reward += self.neg_reward_ia

    def step(self, action):
        """
        Generic step function; takes a step bij calling step_env()
        observation is returned
        Reward is calculated
        Tracers are logged
        Termination case is determined

        returns state, reward, terminate, {}
        """
        self.reward = 0
        if action == self.previous_action:
            self.same_action_count += 1
        elif action != self.previous_action:
            self.same_action_count = 0
        self.previous_action = action

        self.steps += 1
        if action == 0:
            # do nothing but step env
            self.next_O, self.next_D = 0, 0
            self.step_env()

        elif action == 1:
            self.next_O, self.next_D = 1, 1
            self.step_env()

        elif action == 2:
            self.next_O, self.next_D = 2, 1
            self.step_env()


        # Calculate rewards
        # TODO - Reward each pipeline amount for each queue (should be variable with lenght of the pipe)

        # rewards for balancing the pipeline
        for loc in self.diverter_locations:
            amount_in_pipe = len([item for item in  self.items_on_conv if item[2]==self.diverter_locations.index(loc)+1])
            lowerbound = self.ranges[self.diverter_locations.index(loc)][0]
            upperbound = self.ranges[self.diverter_locations.index(loc)][1]
            if amount_in_pipe < lowerbound:
                self.reward += self.shortage_reward
            elif amount_in_pipe > upperbound:
                self.reward += self.flooding_reward
            elif amount_in_pipe > 1.5*upperbound:
                self.reward += self.flooding_reward * 1.5
            elif amount_in_pipe > 2*upperbound:
                self.reward += self.flooding_reward * 2

        # rewards for the queue
        for item in self.in_queue:
            if len(item) < 2:
                self.reward += self.reward_empty_queue

        # rewards for taking cycles in the system
        if len([item for item in self.items_on_conv if
                item[0] == [1, 7]]) == 1:  # in case that negative reward is calculated with cycles
            # self.reward += self.negative_reward_for_cycle  # punish if order carriers take a cycle #tag:punishment
            self.cycle_count +=1

        # Determine termination cases
        if self.termination_condition == 1:
            if self.demand_queues == [[] * i for i in range(self.amount_of_gtps)]:
                self.terminate = True
        elif self.termination_condition == 2:
            if self.init_queues == [[] * i for i in range(self.amount_of_gtps)]:
                self.terminate = True

        # Terminate for too much steps
        if self.steps > 10000:
            self.terminate = True

        #terminate for too much similar steps
        if self.same_action_count > 500:
            self.terminate = True

        #terminate when too many cycles
        if self.cycle_count > self.max_cycle_count:
            self.terminate = True

        next_state = self.make_observation()
        reward = self.reward
        terminate = self.terminate

        return next_state, reward, terminate, {'items_processed': self.items_processed, 'cycle count': self.cycle_count}

    ################## RENDER FUNCTIONS ################################################################################
    def render_plt(self):
        """Simple render function, uses matplotlib to render the image in jupyter notebook + some additional information on the transition points"""
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

        # ##### TURN OFF FOR FASTER RENDERING #############################################################################################
        # # values of the O_states
        # for item in copy(self.output_locations):
        #     x0 = item[0] * resize_factor + 40
        #     y0 = item[1] * resize_factor + 40
        #     draw.text((x0, y0), '{}'.format(self.O_states[self.output_locations.index(item) + 1]), fill='white',
        #               font=ImageFont.truetype(font='arial', size=10, index=0, encoding='unic', layout_engine=None))
        #
        # draw reward
        x0, y0 = self.diverter_locations[0][0] * resize_factor + 130, self.diverter_locations[0][
            1] * resize_factor + 150
        y1 = y0 + 25
        y2 = y1 + 25
        draw.text((x0, y0), ' Total Reward: {}'.format(self.reward), fill='white',
                  font=ImageFont.truetype(font='arial', size=15, index=0, encoding='unic', layout_engine=None))
        # draw.text((x0, y1), ' Positive Reward: {}'.format(self.step_reward_p), fill='green',
        #           font=ImageFont.truetype(font='arial', size=15, index=0, encoding='unic', layout_engine=None))
        # draw.text((x0, y2), ' Negative Reward: {}'.format(self.step_reward_n), fill='red',
        #           font=ImageFont.truetype(font='arial', size=15, index=0, encoding='unic', layout_engine=None))
        # ###################################################################################################################################

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
        width, height = img.size
        if width > self.render_width:
            new_width = self.render_width
            new_height = int(height * (new_width / width))
            img = img.resize((new_width, new_height), resample=Image.BOX)
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
