import pathlib
import argparse
import rl
from os import listdir
from os.path import join, isfile
from rl.environments import *
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback
from rl.baselines import get_parameters
from rl.helpers import launch_tensorboard
import logging
import yaml



"""
Usage of this trainer:
    python train2.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -n [LOGNAME] -c [CONFIG] -t [TENSORBOARD]
    e.g.
    python train2.py -e TestEnv -s Test123 -n Test123 -c config1 -t 

"""
#CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements #DEBUG: show extra info
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment.')
    parser.add_argument('-s', '--subdir', type=str, help='Subdir to combine and analyze.')
    parser.add_argument('-n', '--name' , type=str, help='Name of the specific model.')
    parser.add_argument('-c', '--config', type=str, help='Name of config file in config/name')
    parser.add_argument('-t', '--tensorboard', action='store_true', help='If you want to run a TB node.')
    args = parser.parse_args()
    path = pathlib.Path().absolute()
    specified_path = join(path, 'rl', 'trained_models', args.environment, args.subdir)


    try:
       config_path = join(path, 'rl', 'config', args.subdir, '{}.yml'.format(args.config))
       with open(config_path) as f:
           config = yaml.safe_load(f)
       print('\nLoaded config file from: {}\n'.format(config_path))

    except:
        print('specified config is not in path, getting original config: {}.yml...'.format(args.environment))
        # load config and variables needed
        config = get_parameters(args.environment)

    pass

    try:
        files_gen = (file for file in listdir(specified_path)
                 if isfile(join(specified_path, file))==False)
        files = [file for file in files_gen]
        max_in_dir = max([int(var[0]) for var in files]) + 1
    except:
        max_in_dir = 0
        print('max dir is {}'.format(max_in_dir))

    #getmodelvars
    model_config = config['models']['PPO2']
    n_steps = config['main']['n_steps']
    save_every = config['main']['save_every']
    n_workers = config['main']['n_workers']
    n_checkpoints = n_steps // save_every

    #load environment with config variables
    env_obj = getattr(rl.environments, args.environment)
    env = env_obj(config)

    # multiprocess environment
    env_8 = make_vec_env(lambda: env, n_envs=n_workers)

    # callback for evaluation
    eval_callback = EvalCallback(env, best_model_save_path=join(specified_path, 'Bestmodel_{}'.format(args.name)),
                                 log_path=specified_path, eval_freq=10000,
                                 n_eval_episodes = 10, verbose=1,
                                 deterministic=False, render=False)

    #train model
    try:
        try:
            model_path = join(specified_path, 'best_model_x.zip')
            model = PPO2.load(model_path, env=env_8, tensorboard_log=specified_path)
            #model = PPO2('MlpPolicy', env=env_8, tensorboard_log=specified_path, **model_config).load(args.modelpath, env=env_8)
            print("model loaded")

        except:
            model = PPO2('MlpPolicy', env=env_8, tensorboard_log=specified_path,  **model_config)
            print('new model created')

        #Launch the tensorboard
        if args.tensorboard:
            launch_tensorboard(specified_path)

        for i in range(n_checkpoints):
            model.learn(total_timesteps=save_every, tb_log_name='{}_{}'.format(max_in_dir, args.name),  callback=eval_callback)
            model_path = join(specified_path, '{}_model_{}_{}.zip'.format(max_in_dir,args.name, i+1))
            model.save(model_path)

        # save the config file in the path, for documentation purposes
        print('Saving the config file in path: {}'.format(specified_path))
        with open(join(specified_path,'Bestmodel_{}'.format(args.name), 'config.yml'), 'w') as f:
            yaml.dump(config, f, indent=4, sort_keys=False, line_break=' ')
    except KeyboardInterrupt:
        print('Saving model . .                                    ')
        model_path = join(specified_path, '{}_model_{}_{}_interupt.zip'.format(max_in_dir, args.name, i+1))
        model.save(model_path)

        # save the config file in the path, for documentation purposes
        print('Saving the config file in path: {}'.format(specified_path))
        with open(join(specified_path,'Bestmodel_{}'.format(args.name), 'config.yml'), 'w') as f:
            yaml.dump(config, f, indent=4, sort_keys=False, line_break=' ')
        print('Done.')