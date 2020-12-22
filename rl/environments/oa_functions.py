import numpy as np
import random
import logging
import copy

#### Function to do warm-start #########################################################################################
def warm_start(self):
    # add items to queues, so queues are not empty when starting with training (empty queue is punished with -1 each timestep)
    for _ in self.operator_locations:
        self.in_queue[self.operator_locations.index(_)].append(
            self.init_queues[self.operator_locations.index(_)][0])

        self.update_init(self.operator_locations.index(_))

        # add to items_on_conv
        self.items_on_conv.append(
            [_, self.queue_demand[self.operator_locations.index(_)], self.operator_locations.index(_) + 1])
        update_queue_demand(self)

# #### Generate the visual conveyor ##################################################################################
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

def to_binary(self, var_in):
    bin_list = []
    binstring = "{0:07b}".format(var_in)
    for var in binstring:
        bin_list.append(int(var))
    return bin_list

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