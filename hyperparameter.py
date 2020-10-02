#from rl.environments.simple_conveyor_9 import simple_conveyor_9
#import yaml
import ray
from ray import tune

# config_path = 'rl/config/simple_conveyor_9.yml'
# with open(config_path, 'r') as f:
#     config = yaml.load(f)
# env = simple_conveyor_9(config)

ray.init(ignore_reinit_error=True)
config = {
    'env': 'CartPole-v0'
}
stop = {
    'timesteps_total': 10000
}
results = tune.run(
    'PPO', # Specify the algorithm to train
    config=config,
    stop=stop
)