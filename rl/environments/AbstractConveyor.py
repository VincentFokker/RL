########################################################################################
# New version with right reward reset build in.
import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import copy, deepcopy
import random
import logging
import gym
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

# TODO: resize the observation space
# TODO: larger warm start
# TODO: Revise the observation

#CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements
logging.basicConfig(level=logging.INFO)

class AbstractConveyor(gym.Env):

######## INITIALIZATION OF VARIABLES ###############################################################################################################
    def __init__(self, config, **kwargs):
        """initialize states of the variables, the lists used"""
        # init config
        self.config = config['environment']
        self.amount_of_gtps = self.config['amount_of_gtps']
        self.amount_of_outputs = self.config['amount_of_outputs']
        self.gtp_demand_size = self.config['gtp_demand_size']
        self.process_time_at_GTP = self.config['process_time_at_GTP']
        self.gtp_buffer_length = self.config['gtp_buffer_length']
        self.pipeline_length = self.config['pipeline_length']
        self.observation_shape = self.config['observation_shape']
        self.in_que_observed = self.config['in_que_observed']
        self.exception_occurence = self.config['exception_occurence']
        self.termination_condition = self.config['termination_condition']
        self.max_items_processed = self.config['max_items_processed']
        self.speed_improvement = self.config['speed_improvement']
        self.empty_env = self.image = self.generate_env()
        self.pallette = (np.asarray(sns.color_palette("Reds", self.amount_of_outputs)) * 255).astype(int)
        self.window_name = 'Conveyor Render v5.1'
        self.render_width = self.config['render_width']
        self.max_cycle_count = self.config['max_cycle_count']
        self.steps_by_heuristic = self.config['steps_by_heuristic']
        self.repurpose_goal = self.config['repurpose_goal']

        #init variables
        self.episode = 0
        self.reward = 0
        self.terminate = False
        self.items_on_conv = []
        self.amount_of_orders_processed = 0
        self.condition_to_transfer = False
        self.condition_to_process = False
        self.idle_time_delta = 0
        self.cycle_count_delta = 0

        #init reward vars
        self.idle_time_reward_factor = self.config['idle_time_reward_factor']
        self.cycle_count_reward_factor = self.config['cycle_count_reward_factor']
        self.output_priming_reward = self.config['output_priming_reward']

        #define locations
        self.diverter_locations =   [[i, 7] for i in range(4, self.amount_of_gtps * 4 + 1, 4)][::-1]
        self.merge_locations =      [[i - 1, 7] for i in range(4, self.amount_of_gtps * 4 + 1, 4)][::-1]
        self.output_locations = [[i + self.diverter_locations[0][0] + self.pipeline_length, 7] for i in
                              range(0, self.amount_of_outputs*2, 2)]
        self.operator_locations = [[i, self.empty_env.shape[0] - 3] for i in range(4, self.amount_of_gtps * 4 + 1, 4)][
                                  ::-1]

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(self.amount_of_outputs * self.amount_of_gtps + 1)

        # determination of observation_space:
        self.shape = 0
        if 1 in self.observation_shape:
            self.shape += 2 * ((self.amount_of_gtps * 4) + self.pipeline_length -1 +2 * self.amount_of_outputs)
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
            self.shape += 4
        if 12 in self.observation_shape:
            self.shape += self.amount_of_gtps * self.amount_of_outputs

        self.observation_space = gym.spaces.Box(shape=(self.shape,),
                                                high=1, low=0,
                                                dtype=np.float)



        #tracers
        self.steps = 0
        self.items_processed = 0
        self.cycle_count = 0

        #init demands
        self.queues = [[random.randint(1, self.amount_of_outputs) for _ in range(self.gtp_demand_size)] for item in range(self.amount_of_gtps)]
        self.init_queues = copy(self.queues)
        self.demand_queues = copy(self.queues)
        self.in_queue = [[] for _ in range(len(self.queues))]
        self.in_pipe = [[] for _ in range(len(self.queues))]
        self.queue_demand = [item[0] for item in self.init_queues]



        ######## Do a warm_start
        self.do_warm_start(int(0.5*self.gtp_buffer_length))

        ####### FOR SIMULATION ONLY
        # a counter to record the processing time at GtP station
        self.W_times = {}
        for i in range(1, len(self.init_queues) + 1):
            try:
                self.W_times[i] = int(self.process_time_at_GTP *(1-self.speed_improvement)) if self.in_queue[i - 1][0] == 1 else int(self.process_time_at_GTP * 3 *(1-self.speed_improvement)) if \
                self.in_queue[i - 1][0] == 2 else int(self.process_time_at_GTP * 6 *(1-self.speed_improvement)) if self.in_queue[i - 1][
                                                                                      0] == 1 else int(self.process_time_at_GTP * 9 *(1-self.speed_improvement))
            except:
                self.W_times[i] = 6
        self.idle_times_operator = {}
        for i in range(len(self.operator_locations)):
            self.idle_times_operator[i] = 0

        self.do_warm_start2(self.steps_by_heuristic)

    def do_warm_start2(self, y):

        for _ in range(y):
            self.do_heuristic_guided_step()

    def do_warm_start(self, x):
        for _ in range(x):
            self.warm_start()
        #self.spawn_item_conv()


    def spawn_item_conv(self):
        low, high = self.diverter_locations[-1], self.output_locations[-1]
        low1, high1 = self.diverter_locations[0], self.output_locations[0]
        cand = [item[:2] for item in self.init_queues][::-1]
        cand = [[item[i] for item in cand] for i in range(2)]
        to_add = [item for sublist in [
            [[[low[0] + random.randint(0, 1) + ((idx2) * 4) + (idx1 * (low1[0])), low[1]], var, self.amount_of_gtps - idx2] for
             idx2, var in enumerate(item)] for idx1, item in enumerate(cand)] for item in sublist]
        to_add = [item for item in to_add if item[0][0] + 1 <= high[0]]
        self.items_on_conv += to_add

    def warm_start(self):
        # add items to queues, so queues are not empty when starting with training (empty queue is punished with -1 each timestep)
        for _ in self.operator_locations:
            self.in_queue[self.operator_locations.index(_)].append(
                self.init_queues[self.operator_locations.index(_)][0])

            self.update_init(self.operator_locations.index(_))

            # add to items_on_conv

            for i in range(0, self.gtp_buffer_length):
                if len(self.in_queue[0]) == i + 1:
                    self.items_on_conv.append(
                        [[_[0], _[1] - i], self.queue_demand[self.operator_locations.index(_)],
                         self.operator_locations.index(_) + 1])
            self.update_queue_demand()

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
            if quenr == i+1:
                self.init_queues[i].append(variable)

    def generate_env(self):
        """returns empty env, with some variables about the size, lanes etc."""
        empty = np.zeros(
            (8 + self.gtp_buffer_length + 2, 4 * self.amount_of_gtps + self.pipeline_length + 2* self.amount_of_outputs + 1,
             3))  # height = 15, width dependent on amount of stations
        for i in range(2, empty.shape[1] - 2):
            empty[2][i] = (255, 255, 255)  # toplane = 2
            empty[7][i] = (255, 255, 255)  # bottom lane = 7
        for i in range(2, 8):
            empty[i][1] = (255, 255, 255)  # left lane
            empty[i][empty.shape[1] - 2] = (255, 255, 255)  # right lane

        for i in range(8, empty.shape[0] -2):
            for j in range(4, self.amount_of_gtps * 4 + 1, 4):  # GTP lanes
                empty[i][j] = (255, 255, 255)  # Gtp in
                empty[i][j - 1] = (250, 250, 250)  # gtp out
            for j in range(empty.shape[1] - self.amount_of_outputs * 2 - 1, empty.shape[1] - 2, 2):  # order carrier lanes
                empty[i][j] = (255, 255, 255)
        for i in range(4, self.amount_of_gtps * 4 + 1, 4):  # divert and merge points
            empty[7][i - 1] = (255, 242, 229)  # merge
            empty[7][i] = (255, 242, 229)  # divert

        for i in range(empty.shape[1] - self.amount_of_outputs * 2 - 1, empty.shape[1] - 2, 2):  # output points
            empty[7][i] = (255, 242, 229)

        return empty

    def encode(self, var):
        """encodes categorical variables 0-3 to binary"""
        return (0,0) if var == 0 else (0,1) if var == 1 else (1,0) if var == 2 else (1,1) if var ==3 else var
#####################################################################################################################################
## For Heuristic approach

    def do_heuristic_guided_step(self, bezetgr=15):
        """
        Sets a step in the environment based on the FiFo policy.
        """
        order_sequence = []  # we build a FiFo list here
        for idx, queue in enumerate(self.in_queue[::-1]):  # reversed, because you want to service the last queue first
            if len(queue) + len(self.in_pipe[::-1][idx]) < 0.5 * self.gtp_buffer_length + (self.pipeline_length // bezetgr):
                try:
                    current_demand = [item for item in self.init_queues[::-1]][idx][len(self.in_pipe[::-1][idx]):][0]
                    order_sequence.append((current_demand, len(self.in_queue) - idx))
                except:
                    pass
        try:
            order_type, goal = order_sequence[0]
        except:
            order_type, goal = 0, 0

        self.step(None, order_type, goal)

#############################################################################################################################
    def make_observation(self):
        '''Builds the observation from the available variables
        1. occupation of the output points - {1,0} - occupied or not e.g. - [0, 1, 1]                       +3 = 3 = amount of output
        2. queue demand - {1,0} - one-hot-encoded - 3 types e.g. [3, 2, 2] > [0, 0, 1, 0, 1, 0, 0, 1, 0]    +9 = 12      > Possibly add more demand
        3. amount of items in pipeline - per pipeline / 25 > [0.16, 0.32, 0.08]                             +3 = 15      > could also say: 1-7 = x/7, more then 7 = 1 | or a share for the amount vs. demand: in_pipe/demand = 1 (o.i.d)
        4. amount of items in queues - per queue /7 > [0.14285714, 0.14285714, 0.        ]                  +3 = 18
        5. amount of itemst that took a cycle / self.max_amount_of_cycles                                   +1 = 19      > could also say; more then x cycles = 1 e.g. 25
        6. Usability of the pipeline (1var)

        '''
        ### 1 . For the obeservation of the conveyor ########################################################################
        self.carrier_type_map_obs = np.zeros((self.empty_env.shape[0], self.empty_env.shape[1], 1)).astype(float)
        for item in self.items_on_conv:
            self.carrier_type_map_obs[item[0][1]][item[0][0]] = item[1]

        type_map_obs = self.carrier_type_map_obs[2:8, 1:-1]  # cut padding #for the carrier type
        carrier_type_map_obs = type_map_obs[-1]  # Only observe bottom lane #top and bottom lane for the carrier type
        type_map_obs = np.array(
            [self.encode(item) for item in list(carrier_type_map_obs)]).flatten()  # binary encoded memory for the type
        logging.debug(carrier_type_map_obs)
        # TODO: return:type_map_obs

        ###  2. Occupation of the The output points ########################################################################
        output_points = carrier_type_map_obs[-2 * self.amount_of_outputs:][::2]  ## returns: array([[3.],[3.],[3.]])
        output_points = np.array([1 if item != 0 else 0 for item in output_points])  # Returns array(1, 1, 1)
        logging.debug(output_points)
        # TODO: return: output_points

        ### 3. For the observation of the items in queue ##################################################################
        # length of each queue (how full)            #some indicator of how long it takes to process this full queue (consider 1- x)
        in_queue = [len(item) * (1 / self.gtp_buffer_length) for item in self.in_queue]
        in_queue = np.array(in_queue).flatten()

        # TODO: return: in_queue
        ### 4. For the observation of the demand of the GtP Queue #########################################################
        # make the init list
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
        var=0
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
        remaining_processtime = [sum(item) * 1 / (self.amount_of_outputs * 7) for item in self.in_queue]
        remaining_processtime = np.array(remaining_processtime).flatten()

        # TODO: return: remaining_processtime

        ##### 9. Var if queues can still take items ########################################################
        cantake = []
        isempty = []
        for queue in self.in_queue:
            if len(queue) < 7:
                cantake.append(1)
            elif len(queue) == 7:
                cantake.append(0)
            # TODO: return: cantake

            ##### 10. Var if queue is lower then 2 ##################################################################
            if len(queue) < 2:
                isempty.append(1)
            elif len(queue) >= 2:
                isempty.append(0)
        # TODO: return: isempty

        #### 11. amount of items in lead #########################################################################
        bottom_conv = [item[1] for item in self.items_on_conv if item[0][1] == 7]
        info = []
        if 1 in bottom_conv:
            info.append(1)
            info.append(len([item for item in bottom_conv if item == 1]) / (
                    (self.amount_of_gtps * 4) + 6 + 2 * self.amount_of_outputs))
        else:
            info.append(0)
            info.append(0)

        if 2 in bottom_conv:
            info.append(1)
            info.append(len([item for item in bottom_conv if item == 2]) / (
                    (self.amount_of_gtps * 4) + 6 + 2 * self.amount_of_outputs))
        else:
            info.append(0)
            info.append(0)
        info = np.array(info)

        # TODO: return: info

        #### 12. in pipeline for each queue ###########################################################################
        in_pipe = [[len([item for item in self.items_on_conv if item[2] == i and item[1] == j]) for j in
                    range(1, self.amount_of_outputs + 1)] for i in range(1, self.amount_of_gtps + 1)]
        in_pipe = np.array(in_pipe).flatten()
        in_pipe = np.array([1 if item > (self.gtp_buffer_length + self.pipeline_length // 15) else item / (
                    self.gtp_buffer_length + self.pipeline_length // 15) for item in in_pipe])

        # TODO: return:in_pipe
        ### 13. what is currently in pipe ##############################################################################


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
        if 12 in self.observation_shape:
            obs = np.append(obs, in_pipe)

        return obs


 ########################################################################################################################################################
 ## RESET FUNCTION 
 #            
    def reset(self):
        """reset all the variables to zero, empty queues
        must return the current state of the environment"""
        #init variables
        self.episode += 1
        print('Ep: {:5}, steps: {:3}, R: {:3.3f}'.format(self.episode, self.steps, self.reward), end='\r')

        self.reward = 0
        self.terminate = False
        self.items_on_conv = []
        self.amount_of_orders_processed = 0
        self.condition_to_transfer = False
        self.condition_to_process = False
        self.idle_time_delta = 0
        self.cycle_count_delta = 0

        #reset queue demands
        self.queues = [[random.randint(1, self.amount_of_outputs) for _ in range(self.gtp_demand_size)] for item in
                       range(self.amount_of_gtps)]
        self.init_queues = copy(self.queues)
        self.demand_queues = copy(self.queues)
        self.in_queue = [[] for _ in range(len(self.queues))]
        self.in_pipe = [[] for _ in range(len(self.queues))]
        self.queue_demand = [item[0] for item in self.init_queues]

        #reset tracers
        self.items_processed = 0
        self.steps = 0
        self.cycle_count = 0

        ######## Do a warm_start
        self.do_warm_start(int(0.5*self.gtp_buffer_length))

        self.W_times = {}
        for i in range(1, len(self.init_queues) + 1):
            try:
                self.W_times[i] = int(self.process_time_at_GTP*(1-self.speed_improvement)) if self.in_queue[i - 1][
                                                              0] == 1 else int(self.process_time_at_GTP * 3*(1-self.speed_improvement)) if \
                self.in_queue[i - 1][0] == 2 else int(self.process_time_at_GTP * 6*(1-self.speed_improvement)) if self.in_queue[i - 1][
                                                                                      0] == 1 else int(self.process_time_at_GTP * 9*(1-self.speed_improvement))
            except:
                self.W_times[i] = 6
        self.idle_times_operator = {}
        for i in range(len(self.operator_locations)):
            self.idle_times_operator[i] = 0

        return self.make_observation()

########################################################################################################################################################
## PROCESSING OF ORDER CARRIERS AT GTP
# 
    def process_at_GTP(self):
        # for each step; check if it needed to process an order carrier at GTP
        O_locs = deepcopy(self.operator_locations)
        for Transition_point in O_locs:  # For all operator locations, check:

            try:
                if self.demand_queues[O_locs.index(Transition_point)][0] != self.in_queue[O_locs.index(Transition_point)][
                    0]:
                    self.condition_to_transfer = True
                elif self.demand_queues[O_locs.index(Transition_point)][0] == self.in_queue[O_locs.index(Transition_point)][
                    0]:
                    self.condition_to_process = True
            except:
                self.condition_to_transfer = False
                self.condition_to_process = False

            if self.W_times[O_locs.index(Transition_point) + 1] == 0:  # if the waiting time is 0:
                self.idle_times_operator[O_locs.index(Transition_point)] += 1
                logging.debug(
                    'Waiting time at GTP {} is 0, check done on correctness:'.format(O_locs.index(Transition_point) + 1))
                if random.random() < self.exception_occurence:  # if the random occurence is below exception occurence (set in config) do:
                    # remove an order carrier (broken)
                    logging.debug('With a change percentage an order carrier is removed')
                    logging.debug('transition point is: {}'.format(Transition_point))
                    # self.update_queues(O_locs.index(Transition_point)+1, [item[1] for item in self.items_on_conv if item[0] == Transition_point][0])
                    self.W_times[O_locs.index(Transition_point) + 1] = 1
                    # self.O_states[[item[1] for item in self.items_on_conv if item[0] == Transition_point][0]] +=1
                    self.items_on_conv = [item for item in self.items_on_conv if item[0] != Transition_point]

                elif self.condition_to_transfer:
                    # move order carrier back onto system via transfer - merge
                    for item in self.items_on_conv:
                        if item[0] == Transition_point:
                            item[0][0] -= 1
                    self.W_times[O_locs.index(Transition_point) + 1] = 1
                    self.update_queues(O_locs.index(Transition_point) + 1, self.in_queue[O_locs.index(Transition_point)][0])
                elif self.condition_to_process:
                    # Process an order at GTP successfully
                    logging.debug('Demand queues : {}'.format(self.demand_queues))
                    logging.debug('In queue : {}'.format(self.in_queue))
                    logging.debug('items on conveyor : {}'.format(self.items_on_conv))
                    logging.debug('right order carrier is at GTP (location: {}'.format(Transition_point))
                    logging.debug('conveyor memory before processing: {}'.format(self.items_on_conv))
                    self.items_on_conv = [item for item in self.items_on_conv if item[0] != Transition_point]

                    self.items_processed += 1
                    logging.debug('order at GTP {} processed'.format(O_locs.index(Transition_point) + 1))
                    logging.debug('conveyor memory after processing: {}'.format(self.items_on_conv))

                    # when processed, remove order carrier from demand queue
                    try:
                        # remove from demand queue
                        self.demand_queues[O_locs.index(Transition_point)] = self.demand_queues[
                                                                                 O_locs.index(Transition_point)][1:]
                    except:
                        logging.debug("Except: Demand queue for this lane is allready empty")

                    # set new timestep for the next order
                    # try:
                    #     next_type = [item[1] for item in self.items_on_conv if item[0] == [Transition_point[0], Transition_point[1]-1]][0]
                    #
                    # except:
                    #     next_type = 99
                    # self.W_times[O_locs.index(Transition_point)+1] = self.process_time_at_GTP if next_type == 1 else self.process_time_at_GTP*5 if next_type == 2 else self.process_time_at_GTP*10 if next_type == 3 else self.process_time_at_GTP*12 if next_type == 4 else self.process_time_at_GTP*15
                    to_check = self.in_queue[O_locs.index(Transition_point)][:1]
                    next_W_time = 0 if to_check == [] else int(self.process_time_at_GTP*(1-self.speed_improvement)) if to_check == [
                        1] else int(self.process_time_at_GTP * 5*(1-self.speed_improvement)) if to_check == [
                        2] else int(self.process_time_at_GTP * 10*(1-self.speed_improvement)) if to_check == [3] else int(self.process_time_at_GTP * 10*(1-self.speed_improvement))
                    self.W_times[O_locs.index(Transition_point) + 1] = next_W_time
                    self.idle_times_operator[O_locs.index(Transition_point)] -= 1
                    logging.debug('new timestep set at GTP {} : {}'.format(O_locs.index(Transition_point) + 1, self.W_times[
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
        ####make carrier type map (observe the current state again)
        self.carrier_type_map = np.zeros((self.empty_env.shape[0], self.empty_env.shape[1], 1)).astype(float)
        for item in self.items_on_conv:
            self.carrier_type_map[item[0][1]][item[0][0]] = item[1]

        # process orders at GTP
        self.process_at_GTP()

        # do a step for all items on the conveyor
        for item in self.items_on_conv:
            if item[0] in self.diverter_locations:

                # condition 1: if the item at Dloc == the current demand
                condition_1 = item[1] == self.queue_demand[self.diverter_locations.index(item[0])]
                # condition 2: if the goal of the current item == the current gtp queue
                condition_2 = item[2] == (self.diverter_locations.index(item[0]) + 1) or item[2] == 999
                # condition 3: queue is not full
                condition_3 = [item[0][0], item[0][1] + 1] not in [item[0] for item in self.items_on_conv]

                if condition_1 and condition_2 and condition_3:
                    self.update_init(self.diverter_locations.index(item[0]))
                    self.update_queue_demand()
                    self.in_queue[self.diverter_locations.index(item[0])].append(item[1])
                    self.in_pipe[self.diverter_locations.index(item[0])] = self.in_pipe[self.diverter_locations.index(item[0])][1:]
                    #self.reward += self.positive_reward_for_divert + self.diverter_locations.index(item[0]) * 4  # postive_reward_for_divert
                    item[0][1] += 1
                    logging.debug('moved carrier into lane')

                elif condition_2 and condition_3 and not condition_1:
                    #self.reward += self.wrong_sup_at_goal
                    item[0][0] -= 1
                    if self.repurpose_goal:
                        item[2] = 999

                elif condition_1 and condition_2 and not condition_3:
                    #self.reward += self.flooding_reward
                    item[0][0] -= 1
                    if self.repurpose_goal:
                        item[2] = 999

                elif condition_2 and not condition_1 and not condition_3:
                    #self.reward += self.wrong_sup_at_goal + self.flooding_reward
                    item[0][0] -= 1
                    if self.repurpose_goal:
                        item[2] = 999
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
                    self.empty_env.shape[0] - 3 and item[0][0] < self.amount_of_gtps * 4 + 3 and \
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
                self.in_pipe[self.next_D-1].append(self.output_locations.index(loc) + 1)
            if condition1 and condition2 == False:
                pass
                #self.reward += self.neg_reward_ia


    def step(self, action=None, next_O=None, next_D=None):
        """
        Allows to set Next_O and next_D for the heuristic approach.
        Generic step function; takes a step bij calling step_env()
        observation is returned
        Reward is calculated
        Tracers are logged
        Termination case is determined

        returns state, reward, terminate, {}
        """
        self.next_O = next_O
        self.next_D = next_D
        self.reward = 0
        self.steps += 1
        if action == 0:
            # do nothing but step env
            self.next_O, self.next_D = 0, 0

        #for the rest of the actions
        for i in range(1, self.amount_of_outputs * self.amount_of_gtps + 1):
            if action == i:
                self.next_O, self.next_D = (i - 1) // self.amount_of_gtps + 1, ((i - 1) % self.amount_of_gtps) + 1

        #before the step
        Idle_time1 = sum(self.idle_times_operator.values())
        cycle_count1 = self.cycle_count

        self.step_env()

        #after the step
        Idle_time2 = sum(self.idle_times_operator.values())
        self.idle_time_delta = Idle_time2-Idle_time1

        ## cycle tracer
        # rewards for taking cycles in the system
        if len([item for item in self.items_on_conv if
                item[0] == [1, 7]]) == 1:  # in case that negative reward is calculated with cycles
            #self.reward += self.negative_reward_for_cycle  # punish if order carriers take a cycle #tag:punishment
            self.cycle_count +=1

        #after cyclecount
        cycle_count2 = self.cycle_count
        self.cycle_count_delta = cycle_count2 - cycle_count1

        output_reward = 0
        # priming to output carriers
        if self.next_O != 0:
            output_reward = self.output_priming_reward
        step_reward = -1 * (self.idle_time_delta * self.idle_time_reward_factor + self.cycle_count_delta * self.cycle_count_reward_factor)
        self.reward += step_reward + output_reward

        ## terminate on high cycle count
        if self.cycle_count > self.max_cycle_count:
            self.terminate = True

        ## termination conditions
        if self.termination_condition == 1:
            if self.demand_queues == [[] * i for i in range(self.amount_of_gtps)]:
                self.terminate = True
        elif self.termination_condition == 2:
            if self.init_queues == [[] * i for i in range(self.amount_of_gtps)]:
                self.terminate = True
        elif self.termination_condition ==3:
            if self.items_processed > self.max_items_processed:
                self.terminate = True

        next_state = self.make_observation()
        reward = self.reward
        terminate = self.terminate
        return next_state, reward, terminate, {}

   

################## RENDER FUNCTIONS ################################################################################################
    def render_plt(self):
        """Simple render function, uses matplotlib to render the image in jupyter notebook + some additional information on the transition points"""
        image = self.generate_env()

        for item in self.items_on_conv:
            image[item[0][1]][item[0][0]] = np.asarray([self.pallette[0] if item[1] == 1 else self.pallette[1] if
            item[1] == 2 else self.pallette[2] if item[1] == 3 else self.pallette[3]])
        image = image / 255.0
        plt.imshow(np.asarray(image))
        plt.show()


    def render(self, mode='human'):
        """render with opencv, for eyeballing while testing"""
        resize_factor = 36
        box_diameter = 30
        self.image = self.generate_env()
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
            draw.text((x0, y0),'{}'.format(item[2]), fill='black',
                      font=ImageFont.truetype(font='arial', size=20, index=0, encoding='unic', layout_engine=None))

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
            # x2, y2 = item[0] * resize_factor, item[1] * resize_factor - 12
            # x3, y3 = x2 + 10, y2 + 10
            # x4, y4 = x3 + 5, y2
            # x5, y5 = x4 + 10, y4 + 10
            # color1 = 'green' if self.D_condition_1[self.diverter_locations.index(item) + 1] == True else 'red'
            # color2 = 'green' if self.D_condition_2[self.diverter_locations.index(item) + 1] == True else 'red'
            # draw.ellipse([x2, y2, x3, y3], fill=color1, outline=None)
            # draw.ellipse([x4, y4, x5, y5], fill=color2, outline=None)

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

        #resize with PIL
        #img = img.resize((1200,480), resample=Image.BOX)
        width, height = img.size
        if width > self.render_width:
            new_width = self.render_width
            new_height = int(height * (new_width / width))
            img = img.resize((new_width, new_height), resample=Image.BOX)
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

