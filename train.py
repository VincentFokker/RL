import rl.environments
from rl.baselines import Trainer, get_parameters
import os, argparse
import time
import pathlib
import yaml
from os.path import join
"""
A script for training a RL model in a specified environment
A configuration file from ../config/* that corresponds to the name of your environment or the 
environment type.

Usage: python train.py --env_name TestEnv --subdir TestSubdirectory --name NewModel
    or 
       python train.py -e TestEnv -s TestSubdirectory -n NewModel -m DQN
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment. Can be either a any gym environment or a custom one defined in rl.environments')
    parser.add_argument('-s', '--subdir', type=str, help='Subdirectory where the trained model is going to be stored (useful for separating tensorboard logs): e.g. -> ../trained_models/env_type/env/[SUBDIR]/0_model/*')
    parser.add_argument('-n', '--name', type=str, default=None, help='Unique identifier of the model, e.g. -> ../trained_models/env_type/env/subdir/0_[NAME]/*')
    parser.add_argument('-m', '--model', type=str, default=None, help='Reinforcement learning model to use. PPO / ACER / ACKTR / DQN / .')
    parser.add_argument('-c', '--config', type=str, default=None, help='Adusted configuration file located in config/custom folder')
    parser.print_help()
    args = parser.parse_args()
    path = pathlib.Path().absolute()

    trainer = Trainer(args.environment, args.subdir)

    if args.config is not None:
        try:
            config_path = join(path, 'rl', 'config', 'custom', '{}.yml'.format(args.config))
            with open(config_path) as f:
                config = yaml.safe_load(f)
            print('\nLoaded config file from: {}\n'.format(config_path))

        except:
            print('specified config is not in path, getting original config: {}.yml...'.format(args.environment))
            # load config and variables needed
            config = get_parameters(args.environment)
    else:
        config = get_parameters(args.environment)

    if args.model is not None:
        config['main']['model'] = args.model
    trainer.create_model(name=args.name, config_file=config)
    trainer._tensorboard()
    t0 = time.time()
    trainer.train()
    ts = time.time()
    print('Running time for training: {} minutes.'.format((ts-t0)/60))
    #trainer.run(1000)
    trainer._save()