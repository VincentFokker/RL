
import pathlib
import argparse
import rl
from os import listdir
from os.path import join, isfile
from rl.environments import *
from stable_baselines.common import make_vec_env
from rl.baselines.Wrapper import create_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from rl.baselines import *
from rl.helpers import launch_tensorboard
import logging
import yaml

"""
Usage of this trainer:
    python retrain5.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -n [LOGNAME] -c [CONFIG] -t [TENSORBOARD]
    e.g.
    python retrain5.py -e TestEnv -s Test123 -n Test123 -c config1 -t 

"""
# CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements #DEBUG: show extra info
logging.basicConfig(level=logging.INFO)

###### Custom Policies ###########################################################################################
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
        config_path = join(specified_path, 'config.yml'.format(args.config))
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

    # Get the variables of the model
    model_config = config['models']['PPO2']
    model_config['verbose'] = 0
    n_steps = config['main']['n_steps']
    save_every = config['main']['save_every']
    n_workers = config['main']['n_workers']
    policy = config['main']['policy']
    n_checkpoints = n_steps // save_every

    # load environment with config variables
    env_obj = getattr(rl.environments, args.environment)
    env = env_obj(config)

    # multiprocess environment
    env_8 = create_env(args.environment, config=config, n_workers=n_workers)

    # callback for evaluation
    eval_callback = EvalCallback(env, best_model_save_path=specified_path,
                                 log_path=specified_path, eval_freq=10000,
                                 n_eval_episodes=5, verbose=1,
                                 deterministic=False, render=False)

    # train model
    try:
        try:
            model_path = join(specified_path, 'best_model.zip')
            model = PPO2.load(model_path, env=env_8, tensorboard_log=specified_path)
            # model = PPO2('MlpPolicy', env=env_8, tensorboard_log=specified_path, **model_config).load(args.modelpath, env=env_8)
            print("pretrained-model loaded")

        except:
            try:
                model_path = join(specified_path, 'pretrained-model.zip')
                model = PPO2.load(model_path, env=env_8, tensorboard_log=specified_path)
                # model = PPO2('MlpPolicy', env=env_8, tensorboard_log=specified_path, **model_config).load(args.modelpath, env=env_8)
                print("pretrained-model loaded")
            except:
                model = PPO2(policy, env=env_8, tensorboard_log=specified_path, **model_config)
                print('new model created')

        # Launch the tensorboard
        if args.tensorboard:
            launch_tensorboard(specified_path)

        for i in range(n_checkpoints):
            model.learn(total_timesteps=save_every, tb_log_name='{}_{}_{}'.format(max_in_dir, i, args.name),
                        callback=eval_callback)
            model_path = join(specified_path, '{}_model_{}_{}.zip'.format(max_in_dir, args.name, i + 1))
            model.save(model_path)

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
