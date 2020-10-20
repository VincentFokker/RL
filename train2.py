import pathlib
import argparse
from os import listdir
from os.path import join
from rl.environments.SimpleConveyor12 import SimpleConveyor12
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
    args = parser.parse_args()
    path = pathlib.Path().absolute()
    specified_path = join(path, 'rl', 'trained_models', args.environment, args.subdir)

    #load config and variables needed
    config = get_parameters(args.environment)
    model_config = config['models']['PPO2']
    n_steps = config['main']['n_steps']
    save_every = config['main']['save_every']
    n_workers = config['main']['n_workers']
    n_checkpoints = n_steps // save_every

    #load environment with config variables
    env = SimpleConveyor12(config)

    # multiprocess environment
    env_8 = make_vec_env(lambda: env, n_envs=n_workers)

    # callback for evaluation
    eval_callback = EvalCallback(env, best_model_save_path=specified_path,
                                 log_path=specified_path, eval_freq=100000,
                                 deterministic=True, render=False)

    #train model
    try:
        try:
            model = PPO2('MlpPolicy', env=env_8, tensorboard_log=specified_path, **model_config).load(path, env=env_8)
            print("model loaded")
        except:
            model = PPO2('MlpPolicy', env=env_8, tensorboard_log=specified_path,  **model_config)
            print('new model created')

        for i in range(n_checkpoints):
            model.learn(total_timesteps=save_every, tb_log_name=args.name,  callback=eval_callback)
            model_path = join(specified_path, 'model_{}_{}.zip'.format(args.name, i+1))
            model.save(model_path)
    except KeyboardInterrupt:
        print('Saving model . .                                    ')
        model_path = join(specified_path, 'model_{}_{}_interupt.zip'.format(args.name, i+2))
        model.save(model_path)
        print('Done.')