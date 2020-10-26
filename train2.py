import pathlib
import argparse
from os import listdir
from os.path import join, isfile
from rl.environments.SimpleConveyor10 import SimpleConveyor10
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback
from rl.baselines import get_parameters
import logging
from stable_baselines.common.evaluation import evaluate_policy


"""
Usage of this trainer:
    python train2.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -n [LOGNAME]
    e.g.
    python train2.py -e TestEnv -s Test123 -n Test123

"""
#CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements #DEBUG: show extra info
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment.')
    parser.add_argument('-s', '--subdir', type=str, help='Subdir to combine and analyze.')
    parser.add_argument('-n', '--name' , type=str, help='Name of the specific model.')
    parser.add_argument('-m', '--modelpath', type=str, help='modelpath of the previous model')
    args = parser.parse_args()
    path = pathlib.Path().absolute()
    specified_path = join(path, 'rl', 'trained_models', args.environment, args.subdir)
    try:
        files_gen = (file for file in listdir(specified_path)
                 if isfile(join(specified_path, file))==False)
        files = [file for file in files_gen]
        max_in_dir = max([int(var[0]) for var in files]) + 1
    except:
        max_in_dir = 0
        print('max dir is {}'.format(max_in_dir))

    #load config and variables needed
    config = get_parameters(args.environment)
    model_config = config['models']['PPO2']
    n_steps = config['main']['n_steps']
    save_every = config['main']['save_every']
    n_workers = config['main']['n_workers']
    n_checkpoints = n_steps // save_every

    #load environment with config variables
    env = SimpleConveyor10(config)

    # multiprocess environment
    env_8 = make_vec_env(lambda: env, n_envs=n_workers)

    # callback for evaluation
    eval_callback = EvalCallback(env, best_model_save_path=specified_path,
                                 log_path=specified_path, eval_freq=100000,
                                 deterministic=False, render=False)

    #train model
    try:
        try:
            model = PPO2.load(args.modelpath, env=env_8)
            #model = PPO2('MlpPolicy', env=env_8, tensorboard_log=specified_path, **model_config).load(args.modelpath, env=env_8)
            print("model loaded")

        except:
            model = PPO2('MlpPolicy', env=env_8, tensorboard_log=specified_path,  **model_config)
            print('new model created')

        for i in range(n_checkpoints):
            model.learn(total_timesteps=save_every, tb_log_name='{}_{}'.format(max_in_dir, args.name),  callback=eval_callback)
            model_path = join(specified_path, '{}_model_{}_{}.zip'.format(max_in_dir,args.name, i+1))
            model.save(model_path)
    except KeyboardInterrupt:
        print('Saving model . .                                    ')
        model_path = join(specified_path, '{}_model_{}_{}_interupt.zip'.format(max_in_dir, args.name, i+1))
        model.save(model_path)
        print('Done.')