import pathlib
import argparse
from os import listdir
from os.path import isfile, join
import logging
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2
from rl.environments import *
from stable_baselines.common.vec_env import DummyVecEnv
import yaml, rl

"""
Usage of this tester:
    python test2.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -n [RUNNR]
    e.g.
    python test2.py -e TestEnv -s Test123 -n 0

"""
#CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements #DEBUG: show extra info
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment')
    parser.add_argument('-s', '--subdir', type=str, help='Subdir to combine and analyze')
    parser.add_argument('-n', '--num_episode', type=int, help='NUmber of episodes')
    parser.add_argument('-d','--deterministic', action='store_true', help='Deterministic or not')
    args = parser.parse_args()
    path = pathlib.Path().absolute()
    specified_path = join(path, 'rl', 'trained_models', args.environment, args.subdir)
    files_gen = (file for file in listdir(specified_path)
             if isfile(join(specified_path, file)))
    files = [file for file in files_gen]

    # load config and variables needed
    with open(join(specified_path, 'config.yml'), 'r') as c:
        config = yaml.load(c)
        print('\nLoaded config file from: {}\n'.format(specified_path))
    model_config = config['models']['PPO2']

    logs = []


    path = join(specified_path, 'best_model.zip')
    print(path)

    #make env
    # load environment with config variables
    env_obj = getattr(rl.environments, args.environment)
    env = env_obj(config)

    #model = PPO2('MlpPolicy', env=env, tensorboard_log=specified_path, **model_config).load(path, env=env)
    model = PPO2.load(path, env=DummyVecEnv([lambda: env]))

    #evaluate
    result = {}
    results['idle_time'] = 0
    results['cycle_count'] = 0
    results['steps'] = 0
    results['items_processed'] = 0
    for episode in range(args.num_episodes):
        # Run an episode
        state = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, tc = env.step(action)
        results['idle_time'] += sum(env.idle_times_operator.values())
        results['cycle_count'] += env.cycle_count
        results['steps'] += env.steps
        results['items_processed'] += env.items_processed
    results['idle_time'] = results['idle_time'] / episodes
    results['cycle_count'] = results['cycle_count'] / episodes
    results['steps'] = results['steps'] / episodes
    results['items_processed'] = results['items_processed'] / episodes

    print('idle_time: \t {}'.format(results['idle_time'] ))
    print('cycle_count: \t {}'.format(results['cycle_count']))
    print('steps: \t {}'.format(results['steps']))
    print('items_processed: \t {}'.format(results['items_processed']))