from stable_baselines.common.env_checker import check_env
from rl.baselines import get_parameters
import argparse
import rl

"""
Function to check the validity of an evironment according to Stable-baselines format

params-in: 'Environment name'
run as:

    python check_env.py -e [ENVIRONMENT]

"""


if __name__ == "__main__":
    # parse the arguments from the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment.')
    args = parser.parse_args()

    config = get_parameters(args.environment)
    env_obj = getattr(rl.environments, args.environment)
    env = env_obj(config)

    # It will check your custom environment and output additional warnings if needed
    check_env(env)