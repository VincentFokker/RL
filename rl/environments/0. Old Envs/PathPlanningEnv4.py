import copy
import json
import os
from abc import ABC

import cv2
import gym
import numpy as np
import random

# actions space has information on options (dummies for impossible actions) and wether the next states are
# reserved or not by another agv

class PathPlanningEnv4(gym.Env, ABC):

    def __init__(self, config,
                 input_file1="JSON layouts\Custom layout 2 - verkleind - dijkstra - 25.json",
                 input_file2="JSON layouts\TestCases\Custom layout 2-40-20-10000 - 50-50.json", **kwargs):

        # Load in files:
        # - file 1 contains segment information (processed layout),
        # - file 2 contains training situations

        this_file_path = os.path.dirname(os.path.realpath(__file__))
        file_name1 = os.path.join(this_file_path, input_file1)
        with open(file_name1, "r") as file:
            self.segment_data = json.load(file)

        file_name2 = os.path.join(this_file_path, input_file2)
        with open(file_name2, "r") as file:
            self.training_situations = json.load(file)

        # Initialize parameters
        self.config = config['environment']
        self.number_of_AGVs = self.config['number_of_AGVs']  # equal to nr of AGVs in training situations
        self.time_steps_whca = self.config['time_steps_whca']  # equal to time steps in training situations
        self.max_number_of_actions = self.config['max_number_of_actions']  # max number of actions, depends on layout
        self.state_observation_length = self.config['state_observation_length']  # nr of steps look ahead in reservation

        # Rewards
        self.invalid_chosen_reward = self.config['invalid_chosen_reward']
        self.blocked_position_chosen_reward = self.config['blocked_position_chosen_reward']
        self.decision_to_stay_idle_reward = self.config['decision_to_stay_idle_reward']
        self.reward_distance_improvement = self.config['reward_distance_improvement']
        self.reward_distance_improvement_exponential = self.config['reward_distance_improvement_exponential']
        self.normalize_distances_in_state = self.config['normalize_distances_in_state']

        # Initialisation of AGVs environment tracking lists & dictionaries
        self.learning_AGV = random.choice(range(self.number_of_AGVs))
        self.position_information = {}
        self.goal_destinations = {}
        self.nodes_blocked = []
        self.blocking_information = {}
        self.blocked_by_matrix = []
        self.position_information_whca = {}
        self.blocking_information_whca = {}
        self.goal_destinations_whca = {}
        self.possible_actions = []
        self.state = []

        # Gym-related part
        self.r = 0  # Total episode reward
        self.done = False  # Termination
        self.episode = 0  # Episode number
        self.learning_step = 1  # Current planning step of learning AGV, initiated at 1

        # Trackers
        self.choose_blocked_node = 0

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(self.max_number_of_actions)
        # self.observation_space = gym.spaces.Box(shape=(5,), high=10, low=0, dtype=np.uint8)

        # TODO: explain observation space here !!
        self.observation_space = gym.spaces.Box(shape=(self.max_number_of_actions, 2),
                                                high=1, low=0,
                                                dtype=np.uint8)

    def make_observation(self):
        """
        Returns the environment's current state
        """

        # goal destination for current step = to the goal destination of previous step
        self.goal_destinations_whca['whca_%s' % self.learning_step][str(self.learning_AGV)] = \
            self.goal_destinations_whca['whca_%s' % (self.learning_step - 1)][str(self.learning_AGV)]

        current_position = self.position_information_whca['whca_%s' % (self.learning_step - 1)][str(self.learning_AGV)]
        current_goal = self.goal_destinations_whca['whca_%s' % self.learning_step][str(self.learning_AGV)]
        current_distance_from_goal = self.getDijkstraDistance(current_position, current_goal)

        # Start construction of state representation (in a nested list)
        state_observation = [[] for _ in range(self.max_number_of_actions)]  #  [[], [], [], []]
        possible_actions = self.getPossibleActionsReducedMap(current_position)   # possible actions are the next nodes that can be visited
        distances_of_possible_actions = self.getDijkstraDistancesForActionSpace(possible_actions, current_goal)
        distances_of_possible_actions, possible_actions = zip(*sorted(zip(distances_of_possible_actions, possible_actions)))
        possible_actions = list(possible_actions)
        possible_actions.insert(0, current_position)

        # append the next possible nodes to the state observation stace
        for i in range(len(possible_actions)):
            state_observation[i].append(possible_actions[i])
            # for each possible direction to travel to, fill observation with best possible nodes that follow after
            for j in range(self.state_observation_length - 1):
                state_observation[i].append(
                    self.getBestNextState(state_observation[i][-1],
                                          self.goal_destinations_whca['whca_%s' % self.learning_step][
                                              str(self.learning_AGV)]))
        nodes_state_observation = copy.deepcopy(state_observation)
        # example state observation [[4415, 4426, 469, 3915, 4228, 4253, 4278, 4303, 4328, 753], [4426, 469, 3915,
        #                                                              4228, 4253, 4278, 4303, 4328, 753, 1281], [], []]

        # check if the added positions will be occupies at time agv could reach this position.
        # if occupied, spot in state == 1, else == 0.
        for i in range(len(possible_actions)):
            for j in range(self.state_observation_length):
                if int(j + self.learning_step) <= int(self.time_steps_whca):
                    state_observation[i][j] = self.checkForReservationAtTime(self.learning_AGV, state_observation[i][j],
                                                                             self.number_of_AGVs,
                                                                             self.position_information_whca[
                                                                                 'whca_%s' % (
                                                                                         self.learning_step + j)])
                if int(j + self.learning_step) > int(self.time_steps_whca):
                    state_observation[i][j] = 0

            # get normalised change in distance valueand add to the state observation at first spot in the sublist
            if self.normalize_distances_in_state is True:
                change_in_distance = current_distance_from_goal - self.getDijkstraDistance(nodes_state_observation[i][0], self.goal_destinations_whca['whca_%s' % self.learning_step][str(self.learning_AGV)])

                if 0 < change_in_distance <= 25:
                    distance_value = 1
                if change_in_distance == 0:
                    distance_value = 0
                if -25 <= change_in_distance < 0:
                    distance_value = -1
                if -100 <= change_in_distance < -25:
                    distance_value = -2
                if -250 <= change_in_distance < -100:
                    distance_value = -3
                if change_in_distance < -250:
                    distance_value = -4

                state_observation[i].insert(0, distance_value)  # insert at index 0
        # state_observation looks like: [[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [], []]
            # first values in the lists represent the normalized distance value

        # count until a blocked position is found. return the numbers of available next steps in the state observation
        for i in range(len(possible_actions)):
            counter = 0
            counter_list = []
            for j in range(1, self.state_observation_length + 1):
                if state_observation[i][j] == 0:
                    counter += 1
                if state_observation[i][j] != 0 or j == self.state_observation_length:
                    counter_list.append(counter)
            state_observation[i] = [state_observation[i][0], counter_list[0]]
        # state_observation looks like: [[0, 1], [1, 10], [], []]

        # add dummies in the state observation for actions that are not possible
        for i in range(len(possible_actions), self.max_number_of_actions):
            state_observation[i] = [9999] * 2

        # add dummies to action space (used for checking if action is possible)
        while len(possible_actions) != self.max_number_of_actions:
            possible_actions.append('XXX')
        # state observation look like: [[0, 1], [1, 10], [9999, 9999], [9999, 9999]]

        # store the action possibilities
        self.possible_actions = list(possible_actions)
        # used for saving current state only, no RL Requirement!
        self.state = list(state_observation)

        return list(state_observation)

    def reset(self):
        """
        This method gets called when initializing the environment and every time an episode ends
        (termination condition is reached). Takes no arguments and returns the current state (image or a vector)

        In order to speed up training, test cases have been build in advance (partial plannings)
        When reset() is called, a random situation is selected and its current state is returned.
        """
        self.episode += 1
        training_situation = random.choice(range(len(self.training_situations['test_cases'])))  # 174

        # reset position, blocking and goal information to the new training situation
        self.learning_AGV = self.training_situations['test_cases'][training_situation]['learning_AGV']
        self.position_information_whca = self.training_situations['test_cases'][training_situation][
            'position_information_whca']
        self.blocking_information_whca = self.training_situations['test_cases'][training_situation][
            'blocking_information_whca']
        self.goal_destinations_whca = self.training_situations['test_cases'][training_situation][
            'goal_destinations_whca']

        self.learning_step, self.r = 1, 0
        self.done = False
        print('situation_nr:', training_situation, 'goal_destination:', self.goal_destinations_whca['whca_0'][str(self.learning_AGV)])

        # return state observation
        return self.make_observation()

    def step(self, action):
        # translate a action to a step in the reservation system.
        # return the corresponding rewards

        reward = 0  # base reward
        current_position = self.position_information_whca['whca_%s' % (self.learning_step - 1)][str(self.learning_AGV)]
        current_goal = self.goal_destinations_whca['whca_%s' % self.learning_step][str(self.learning_AGV)]
        possible_actions = self.possible_actions
        blocks_for_agv = self.getBlocksForAGV(self.learning_AGV, self.number_of_AGVs,
                                              self.position_information_whca['whca_%s' % self.learning_step])

        # Translate chosen action to desired next node for agv
        desiredNextNode = possible_actions[int(action)]

        # if actions is chosen to remain at the current position, return corresponding reward
        if desiredNextNode == current_position:
            reward += self.decision_to_stay_idle_reward

        # if invalid actions is chosen, return corresponding reward:
        if desiredNextNode == 'XXX':
            reward += self.invalid_chosen_reward

        # if action to travel is chosen (!= current position) but is not available (blocked):
        if self.blockingsDesiredNodeAvailable(desiredNextNode, blocks_for_agv) is False \
                and desiredNextNode != current_position and desiredNextNode != 'XXX':
            reward += self.blocked_position_chosen_reward
            self.choose_blocked_node += 1
            self.r += reward
            state = self.state
            return state, reward, self.done, {}

        # if chosen next node is available and not 'XXX', plan step and return corresponding reward:
        if self.blockingsDesiredNodeAvailable(desiredNextNode, blocks_for_agv) is True and desiredNextNode != 'XXX':
            self.position_information_whca['whca_%s' % self.learning_step][str(self.learning_AGV)] = desiredNextNode
            self.blocking_information_whca['whca_%s' % self.learning_step][
                str(self.learning_AGV)] = self.getBlocksLinkedToNode(
                self.position_information_whca['whca_%s' % self.learning_step][str(self.learning_AGV)])
            if self.reward_distance_improvement: # option to determine reward based on change in distance

                # if rewarding according to absolute change in distance
                if self.reward_distance_improvement_exponential is False and self.normalize_distances_in_state is False:
                    reward += ((self.getDijkstraDistance(current_position, current_goal) - self.getDijkstraDistance(
                        desiredNextNode, current_goal)) / 10)

                # if rewarding according to exponential change in distance
                if self.reward_distance_improvement_exponential is True and self.normalize_distances_in_state is False:
                    reward += (((self.getDijkstraDistance(current_position, current_goal) - self.getDijkstraDistance(
                        desiredNextNode, current_goal)) / 10)**2)

                # if rewarding according to normalized distance values:
                if self.reward_distance_improvement_exponential is False and self.normalize_distances_in_state is True:
                    change_in_distance = (self.getDijkstraDistance(current_position, current_goal) - self.getDijkstraDistance(
                        desiredNextNode, current_goal))
                    if 0 < change_in_distance <= 25:
                        reward += 3.5
                    if change_in_distance == 0:
                        reward += -0
                    if -25 <= change_in_distance < 0:
                        reward += -2.5
                    if -100 <= change_in_distance < -25:
                        reward += -5
                    if -250 <= change_in_distance < -100:
                        reward += -10
                    if change_in_distance < -250:
                        reward += -25

                if self.reward_distance_improvement_exponential is True and self.normalize_distances_in_state is True:
                    return 'REWARDING SETTINGS INVALID'

        # if current position not available and action is feasible (not 'XXX'), there is a conflict in the planning
        elif desiredNextNode == current_position and self.blockingsDesiredNodeAvailable(desiredNextNode, blocks_for_agv) is False and desiredNextNode != 'XXX':
            # find blocking agvs and remove them from the planning
            # NOTE: this approach is only select in learning, can result in init loops in execution

            # find blocking agv(s)
            blocking_agvs = []
            for key, value in self.blocking_information_whca['whca_%s' % self.learning_step].items():
                if [True for i in self.getBlocksLinkedToNode(desiredNextNode) if i in value]:
                    blocking_agvs.append(int(key))

            if len(blocking_agvs) == 0:
                print('ERROR no blocking agvs')
            if self.learning_AGV in blocking_agvs:
                print('ERROR current agv in blocking_agvs')

            # remove blokcing agvs from planning and schedule next step for the learning AGV
            for blocking_agv in blocking_agvs:
                for x in range(1, self.time_steps_whca + 1):
                    self.position_information_whca['whca_%s' % x][str(blocking_agv)] = 0
                    self.blocking_information_whca['whca_%s' % x][str(blocking_agv)] = [0]
                    self.goal_destinations_whca['whca_%s' % x][str(blocking_agv)] = \
                        self.goal_destinations_whca['whca_0'][
                            str(blocking_agv)]
            self.position_information_whca['whca_%s' % self.learning_step][str(self.learning_AGV)] = \
                self.position_information_whca['whca_%s' % (self.learning_step - 1)][
                    str(self.learning_AGV)]
            self.blocking_information_whca['whca_%s' % self.learning_step][
                str(self.learning_AGV)] = self.getBlocksLinkedToNode(
                self.position_information_whca['whca_%s' % (self.learning_step - 1)][str(self.learning_AGV)])

        # return current state if nothing has changed
        state = self.state
        self.r += reward

        # return updated state if something has happened, if nothing has changed the old state is returned
        if desiredNextNode != 'XXX':
            self.learning_step += 1
            state = self.make_observation()

        if self.learning_step == self.time_steps_whca:
            self.done = True

        return state, reward, self.done, {}

    # def render():
    #     pass

    # -- Helper Functions -------------------------------------------------------------------------------------------
    # These functions help to process chosen actions such that a new reservation is made

    def getBlocksLinkedToNode(self, node_id):
        for i in range(len(self.segment_data['blocked_nodes'])):
            if self.segment_data['blocked_nodes'][i]['node_id'] == node_id:
                return self.segment_data['blocked_nodes'][i]['blockings']
        return []

    def getPossibleActionsReducedMap(self, id):
        possible_actions = []
        for i in range(len(self.segment_data['segments'])):
            if self.segment_data['segments'][i]['startNodeId'] == id:
                possible_actions.append(self.segment_data['segments'][i]['endNodeId'])
        return list(set(possible_actions))

    def getBlocksForAGV(self, agv_id, number_of_AGVs, position_information):
        nodes_blocked = []
        list_of_other_AGVs = list(range(number_of_AGVs))
        list_of_other_AGVs.remove(agv_id)
        for i in list_of_other_AGVs:
            if self.getBlocksLinkedToNode(position_information[str(i)]) is not None:
                nodes_blocked.extend(self.getBlocksLinkedToNode(position_information[str(i)]))
        return nodes_blocked

    def getDijkstraDistance(self, position, goal):
        for i in range(len(self.segment_data["dijkstra distances"])):
            if self.segment_data['dijkstra distances'][i]['position_id'] == position and \
                    self.segment_data['dijkstra distances'][i]['goal_id'] == goal:
                return self.segment_data['dijkstra distances'][i]['min_distance']

    def getDijkstraDistancesForActionSpace(self, lijst, goal):
        distance_lijst = []
        for i in lijst:
            distance_lijst.append(self.getDijkstraDistance(i, goal))
        return distance_lijst

    def getBestNextState(self, id, goal):
        nextPossibleStates = self.getPossibleActionsReducedMap(id)
        # print('nextPossibleStates', id, goal, nextPossibleStates)
        distancesAwayFromGoal = self.getDijkstraDistancesForActionSpace(nextPossibleStates, goal)
        desiredNextNode = nextPossibleStates[distancesAwayFromGoal.index(min(distancesAwayFromGoal))]
        return desiredNextNode

    def checkForReservationAtTime(self, agv, desired_position, number_of_AGVs, position_information):
        blocks = self.getBlocksForAGV(agv, number_of_AGVs, position_information)
        if self.blockingsDesiredNodeAvailable(desired_position, blocks) is False:
            return 1
        if self.blockingsDesiredNodeAvailable(desired_position, blocks) is True:
            return 0

    def getPreviousNodesReducedMap(self, position):
        previous_ids = []
        for i in range(len(self.segment_data['segments'])):
            if self.segment_data['segments'][i]['endNodeId'] == position:
                previous_ids.append(self.segment_data['segments'][i]['startNodeId'])
        return list(set(previous_ids))

    def blockingsDesiredNodeAvailable(self, node, blockings_list):
        blockings_linked_to_node = self.getBlocksLinkedToNode(node)
        result = True
        for x in blockings_linked_to_node:
            for y in blockings_list:
                if x == y:
                    result = False
                    return result

        return result

        # ------------------------------------------------------------------------------------------------------

    def run(self, model, episodes=100):
        """
        Use a trained model to select actions
        """

        try:
            for episode in range(episodes):
                self.done, step = False, 0
                state = self.reset()
                print(episode)
                # print(state)
                while not self.done:
                    action = model.model.predict(state)
                    state, reward, self.done, _ = self.step(action[0])
                    print('action: {}, Reward: {:2.3f}, new actions: {}, new state: {}'.format(action, reward,
                                                                                               self.possible_actions,
                                                                                               state))

        except KeyboardInterrupt:
            pass

    def sample(self):
        """
        Sample random actions and run the environment
        """

        for _ in range(100):
            print('episode:', _)
            self.done = False
            state = self.reset()
            print('start state:', state)
            while not self.done:
                action = self.action_space.sample()
                state, reward, self.done, _ = self.step(action)
                print('action: {}, Reward: {:2.3f}, new actions: {}, new state: {}'.format(action, reward,
                                                                                           self.possible_actions,
                                                                                           state))




if __name__ == "__main__":
    from rl.baselines import get_parameters, Trainer
    import rl.environments

    env = PathPlanningEnv4(get_parameters('PathPlanningEnv4'))

    # #SAMPLE RANDOM ACTIONS
    print('Sampling random actions...')
    env.sample()

    # #TRAIN NEW MODEL (DOES NOT SAVE) AND SAMPLE ACTIONS FROM IT
    # model = Trainer('PathPlanningEnv4', 'models').create_model()
    # model._tensorboard()
    # model.train()
    # print('Training done')
    # input('Run trained model (Enter)')
    # env.run(model)

    # #LOAD IN TRAINED MODEL FOR SAMPLING ACTIONS
    # model = Trainer('PathPlanningEnv4', 'train5050').load_model(1)
    # input('Run trained model (Enter)')
    # # env.create_window()
    # env.run(model)