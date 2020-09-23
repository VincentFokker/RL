import numpy as np
from SC_v12 import simple_conveyor_2
import yaml
import random

def single_point_crossover(input_list):
    """Changes a string with one-point cross-over to a new string"""
    i1 = random.randint(1,len(input_list))
    i2 = random.randint(1,len(input_list))
    t1 = input_list[i1]
    t2 = input_list[i2]
    input_list[i1] = t2
    input_list[i2] = t1
    return input_list


config_path = 'rl/config/simple_conveyor_2.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f)
    
#queues = [random.choices(np.arange(1,config['environment']['amount_gtp']+1), [config['environment']['percentage_small_carriers'], config['environment']['percentage_medium_carriers'], config['environment']['percentage_large_carriers']], k=config['environment']['gtp_buffer_size']) for item in range(config['environment']['amount_gtp'])] # generate random queues
queues = [[2, 3, 2, 1, 3, 2, 3, 3, 3, 2], [3, 2, 3, 2, 2, 3, 1, 2, 2, 3], [2, 2, 1, 2, 2, 3, 2, 3, 2, 3]]
print(queues)

env = simple_conveyor_2(config, queues)

order_list = []
for index in range(len(queues[0])):
    order_list.append([item[index] for item in env.queues])

#flat_list = [item for sublist in l for item in sublist]
order_list = [item for sublist in order_list for item in sublist]
print(order_list)
order_list = order_list + 3*len(order_list) * [0]
print(order_list)

env.reset()
for item in order_list:
    env.step(item)
    env.render()
while env.demand_queues != [[] * i for i in range(env.amount_of_gtps)]:
    env.step(0) 
    env.render()
env.negative_reward