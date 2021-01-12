from rl.helpers import Load_data
import pathlib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from os.path import join
import pandas as pd
import yaml
import rl
path = pathlib.Path().absolute()
scalars = ['episode_reward', 'loss/loss']
timevar = 'step'  # wall_time or step

to_combine=[
    ['ConveyorEnv121', '20210113_0500'],  # pipe10
    ['ConveyorEnv121', '20210113_0530'],  # pipe15
    ['ConveyorEnv121', '20210113_0600'],  # pipe20
    ['ConveyorEnv121', '20210113_0630'],  # pipe25
    ['ConveyorEnv121', '20210113_0700'],  # pipe30
    ['ConveyorEnv121', '20210113_0730'],  # pipe35
    ['ConveyorEnv121', '20210113_0800'],  # pipe40
    ['ConveyorEnv121', '20210113_0830'],  # pipe45
    ['ConveyorEnv121', '20210113_0900']   # pipe50
]

if __name__ == "__main__":
    #parse the arguments from the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--terms', type=str, help='Term to identify specific plot')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to test on.')
    args = parser.parse_args()

    combinations = ['pipeline_{}'.format(i) for i in range(10,55,5)]

    results = {}
    for idx, combination in enumerate(to_combine):
        env1, subdir = combination
        # load config and variables needed
        location_path = join(path, 'rl', 'trained_models', env1, subdir)
        with open(join(location_path, 'config.yml'), 'r') as c:
            config = yaml.load(c)
            print('\nLoaded config file from: {}\n'.format(join(location_path, 'config.yml')))
        model_config = config['models']['PPO2']

        #switch termination cases:
        config['environment']['terminate_on_idle'] = False
        config['environment']['alternative_terminate'] = True

        # initialize env with the config file
        env_obj = getattr(rl.environments, env1)
        env = env_obj(config)

        # load best model from path
        model = PPO2.load(join(location_path, 'best_model.zip'), env=DummyVecEnv([lambda: env]))

        results[combinations[idx]] = {}
        results[combinations[idx]]['configuration'] = '{}x{}'.format(config['environment']['amount_of_gtps'],
                                                                     config['environment']['amount_of_outputs'])
        results[combinations[idx]]['gamma'] = config['models']['PPO2']['gamma']
        results[combinations[idx]]['idle_time'] = 0
        results[combinations[idx]]['cycle_count'] = 0
        results[combinations[idx]]['steps'] = 0
        results[combinations[idx]]['items_processed'] = 0
        results[combinations[idx]]['reward'] = 0

        for episode in range(args.num_episodes):
            # Run an episode
            state = env.reset()
            done = False
            while not done:
                action, _ = model.predict(state, deterministic=True)
                state, reward, done, tc = env.step(action)
                results[combinations[idx]]['reward'] += reward
            results[combinations[idx]]['idle_time'] += sum(env.idle_times_operator.values())
            results[combinations[idx]]['cycle_count'] += env.cycle_count
            results[combinations[idx]]['steps'] += env.steps
            results[combinations[idx]]['items_processed'] += env.items_processed

        results[combinations[idx]]['idle_time'] = results[combinations[idx]]['idle_time'] / args.num_episodes
        results[combinations[idx]]['cycle_count'] = results[combinations[idx]]['cycle_count'] / args.num_episodes
        results[combinations[idx]]['steps'] = results[combinations[idx]]['steps'] / args.num_episodes
        results[combinations[idx]]['items_processed'] = results[combinations[idx]]['items_processed'] / args.num_episodes
        results[combinations[idx]]['reward'] = results[combinations[idx]]['reward'] / args.num_episodes

    resultcsv = pd.DataFrame.from_dict(results).T
    resultcsv['idle_percent'] = resultcsv.idle_time / resultcsv.steps
    resultcsv['cycle_percent'] = resultcsv.cycle_count / resultcsv.items_processed
    resultcsv.to_csv('evaluation_results/results_DRL_{}.csv'.format(args.terms))

    print('Results saved to: evaluation_results/results_DRL_{}.csv'.format(args.terms))