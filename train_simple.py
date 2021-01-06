import pathlib
import argparse
import rl
from os import listdir
from os.path import join, isfile
from rl.environments import *
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.policies import *
from stable_baselines.common.vec_env import DummyVecEnv
from rl.baselines import *
from rl.helpers import launch_tensorboard
import logging
import yaml

class CustomMlpPolicy(FeedForwardPolicy):
    """
    A custom MLP policy architecture initializer

    Arguments to the constructor are passed from
    the config file: -> policies: CustomMlpPolicy
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        # net_architecture = config['shared']
        pconfig = config['policies']['CustomMlpPolicy']
        net_architecture = pconfig['shared']
        net_architecture.append(dict(pi=pconfig['h_actor'],
                                     vf=pconfig['h_critic']))
        print('Custom MLP architecture', net_architecture)
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                         net_arch=net_architecture,
                         feature_extraction="mlp", **_kwargs)
#####################################################################################################################


if __name__ == "__main__":
    # Register the policy, it will check that the name is not already taken
    register_policy('CustomMlpPolicy', CustomMlpPolicy)

    #parse the arguments from the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment.')
    parser.add_argument('-s', '--subdir', type=str, help='Subdir to combine and analyze.')
    parser.add_argument('-n', '--name', type=str, help='Name of the specific model.')
    parser.add_argument('-c', '--config', type=str, help='Name of config file in config/name')
    parser.add_argument('-t', '--tensorboard', action='store_true', help='If you want to run a TB node.')
    args = parser.parse_args()
    path = pathlib.Path().absolute()
    specified_path = join(path, 'rl', 'trained_models', args.environment, args.subdir)

    #check if a config file is defined, and if this is available, otherwise load config from config folder.
    try:
        config_path = join(path, 'rl', 'config', 'custom', '{}.yml'.format(args.config))
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print('\nLoaded config file from: {}\n'.format(config_path))

    except:
        print('specified config is not in path, getting original config: {}.yml...'.format(args.environment))
        # load config and variables needed
        config = get_parameters(args.environment)

    #check if there is allready a trained model in this directory
    try:
        files_gen = (file for file in listdir(specified_path)
                     if isfile(join(specified_path, file)) == False)
        files = [file for file in files_gen]
        max_in_dir = max([int(var[0]) for var in files]) + 1
    except:
        max_in_dir = 0
        print('max dir is {}'.format(max_in_dir))

    GAMMA = 0.99375
    policy = config['main']['policy']
    firsttrain = config['main']['firsttrain']
    secondtrain =config['main']['secondtrain']

    #build env
    env_obj = getattr(rl.environments, args.environment)
    env = env_obj(config)
    env_x = DummyVecEnv([lambda: env])

    # callback for evaluation
    eval_callback = EvalCallback(env_x, best_model_save_path=specified_path,
                                 log_path=specified_path, eval_freq=100000,
                                 n_eval_episodes=5, verbose=1,
                                 deterministic=False, render=False)
    #initiate model
    model = PPO2(policy, env=env_x, tensorboard_log=specified_path, gamma=GAMMA, verbose=0)

    #launch tensorboard
    if args.tensorboard:
        launch_tensorboard(specified_path)

    #start learning
    try:

        #learn model without evaluation
        model.learn(total_timesteps=firsttrain, tb_log_name='{}_{}_first'.format(max_in_dir, args.name))

        #save firsttrain
        model.save(join(specified_path, '{}_firsttrain.zip'.format(max_in_dir)))

        #learn model with evaluations
        model.learn(total_timesteps=secondtrain, tb_log_name='{}_{}_eval'.format(max_in_dir, args.name),
                    callback=eval_callback)

        model.save(join(specified_path, '{}_secondtrain.zip'.format(max_in_dir)))

        # save the config file in the path, for documentation purposes
        print('Saving the config file in path: {}'.format(specified_path))
        with open(join(specified_path, 'config.yml'), 'w') as f:
            yaml.dump(config, f, indent=4, sort_keys=False, line_break=' ')

    except KeyboardInterrupt:
        print('Saving model . .                                    ')
        model_path = join(specified_path, '{}_model_{}_{}_interupt.zip'.format(max_in_dir, args.name, i + 1))
        model.save(model_path)

        # save the config file in the path, for documentation purposes
        print('Saving the config file in path: {}'.format(specified_path))
        with open(join(specified_path, 'config.yml'), 'w') as f:
            yaml.dump(config, f, indent=4, sort_keys=False, line_break=' ')
        print('Done.')