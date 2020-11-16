import argparse, rl
from rl.baselines import get_parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment.')
    args = parser.parse_args()

    config = get_parameters(args.environment)
    env_obj = getattr(rl.environments, args.environment)
    env= env_obj(config)
    env.reset()