import pathlib
import argparse
from os import listdir
from os.path import isfile, join
import logging
import pickle
import yaml
import importlib.util
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import ppo2

"""
Usage of this tester:
    python Test_metric.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -n [RUNNR]
    e.g.
    python Test_metric.py -e TestEnv -s Test123 -n 0

"""
#CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements #DEBUG: show extra info
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment')
    parser.add_argument('-s', '--subdir', type=str, help='Subdir to combine and analyze')
    parser.add_argument('-n', '--name', type=str, help='Which number you want to combine')
    args = parser.parse_args()
    path = pathlib.Path().absolute()
    specified_path = join(path, 'rl', 'trained_models', args.environment, args.subdir)
    model_name = [file for file in listdir(specified_path) if file.startswith(args.name)][0]
    print('Selected model is: {}'.format(model_name))
    path_to_model = join(specified_path, model_name, 'model.zip')
    print('Path to model is: {}'.format(path_to_model))
    path_to_config = join(specified_path, model_name, 'config.yml')
    print('Path to config is: {}'.format(path_to_config))

    #load model
    model = Trainer(args.environment, args.subdir).load_model(args.name)

    #load environment
    env = model = Trainer(args.environment, args.subdir).load_model(args.name)

    #evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print('Mean Reward : {}, Standardized Reward : {}'.format(mean_reward, std_reward))