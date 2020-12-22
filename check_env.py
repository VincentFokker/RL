from stable_baselines.common.env_checker import check_env
from rl.baselines import get_parameters
from rl.environments.SimpleConveyor10 import simple_conveyor_10

config = get_parameters('simple_conveyor_10')
env = simple_conveyor_10

# It will check your custom environment and output additional warnings if needed
check_env(env)