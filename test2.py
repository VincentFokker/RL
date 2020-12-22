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
    parser.add_argument('-r', '--render', action='store_true', help='Render the agents.')
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

    for file in files:
        try:
            path = join(specified_path, file)
            print(path)

            #make env
            # load environment with config variables
            env_obj = getattr(rl.environments, args.environment)
            env = env_obj(config)


            env = DummyVecEnv([lambda: env])
            #model = PPO2('MlpPolicy', env=env, tensorboard_log=specified_path, **model_config).load(path, env=env)
            model = PPO2.load(path, env=env)

            #evaluate
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, render=args.render, deterministic=args.deterministic)
            log = '{}: Mean Reward : {:5.3f}, Standard deviation Reward : {:5.3f}'.format(file, mean_reward, std_reward)
            print(log)
            logs.append(log)
        except:
            print('exception occurred')

    with open(join(specified_path, 'evaluation_log.txt'), 'w') as f:
        for item in logs:
            f.write("%s\n" % item)