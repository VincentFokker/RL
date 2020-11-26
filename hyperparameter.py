import numpy as np
import pathlib
from os.path import join, isfile
from hyperspace import hyperdrive
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.vec_env import DummyVecEnv
import datetime
import os
import yaml
import rl

env_name = 'AbstractConveyor1'


path = pathlib.Path().absolute()


def objective(params):
    """
    Objective function to be minimized.

    Parameters
    ----------
    * params [list, len(params)=n_hyperparameters]
        Settings of each hyperparameter for a given optimization iteration.
        - Controlled by hyperspaces's hyperdrive function.
        - Order preserved from list passed to hyperdrive's hyperparameters argument.
     """
    config_path = join(path, 'rl', 'config', '{}.yml'.format(env_name))
    with open(config_path) as f:
        config = yaml.safe_load(f)
        print('model loaded from path: {}'.format(config_path))
    
    #set the parameters
    prfd, wsag, fr, nria, nrfeq, nrfc = params
    config['environment']['positive_reward_for_divert'] = prfd
    config['environment']['wrong_sup_at_goal'] = wsag
    config['environment']['flooding_reward'] = fr
    config['environment']['neg_reward_ia'] = nria
    config['environment']['negative_reward_for_empty_queue'] = nrfeq
    config['environment']['negative_reward_for_cycle'] = nrfc
    
    print('Current settings for the config: \n\npositive_reward_for_divert \t:\t{}\nwrong_sup_at_goal\t\t:\t{}\n\
flooding_reward\t\t\t:\t{}\nneg_reward_ia\t\t\t:\t{}\nnegative_reward_for_empty_queue\t:\t{}\n\
negative_reward_for_cycle\t:\t{}\n'.format(prfd, wsag, fr, nria, nrfeq, nrfc))
    
    #GET MODEL CONFIG
    model_config = config['models']['PPO2']
    policy = config['main']['policy']
    n_workers = config['main']['n_workers']
    n_steps = config['main']['n_steps']
    n_eval = (n_steps / 8)/10
    
    # load environment with config variables
    env_obj = getattr(rl.environments, env_name)
    env = env_obj(config)
    
    # multiprocess environment
    env_8 = make_vec_env(lambda: env, n_envs=n_workers)
    
    #define folder and path
    now = datetime.datetime.now()
    folder ='{}{}{}_{}{}'.format(now.year, str(now.month).zfill(2), str(now.day).zfill(2), str(now.hour).zfill(2), str(now.minute).zfill(2))
    specified_path = join(path, 'rl', 'trained_models', env_name, 'hyper-parameter', '{}-{}{}{}{}{}{}'.format(folder, prfd, wsag, fr, nria, nrfeq, nrfc))
    print('Results stored in: {}'.format(specified_path))
    
    # callback for evaluation
    eval_callback = EvalCallback(env, best_model_save_path=specified_path,
                                 log_path=specified_path, eval_freq=n_eval,
                                 n_eval_episodes=5, verbose=0,
                                 deterministic=False, render=False)

    model = PPO2(policy, env=env_8, tensorboard_log=specified_path, **model_config)
    
    #LEARN MODEL
    model.learn(total_timesteps=n_steps, tb_log_name='{}_{}_{}_{}_{}_{}'.format(prfd, wsag, fr, nria, nrfeq, nrfc),
                        callback=eval_callback)
    model_path = join(specified_path, 'model_{}_{}_{}_{}_{}_{}.zip'.format(prfd, wsag, fr, nria, nrfeq, nrfc))
    model.save(model_path)
    
    #test
    best_modelpath = join(specified_path, 'best_model.zip')
    test_model = PPO2.load(best_modelpath, env=DummyVecEnv([lambda: env]))
    
    #run test of the model
    episodes = 10
    results = {}
    results['cycle_count'] = 0
    results['idle_time'] = 0
    for episode in range(episodes):
        # Run an episode
        state = env.reset()
        done = False
        meta_data = []
        while not done:
            action, _ = test_model.predict(state, deterministic=False)
            state, reward, done, _ = env.step(action)
            if done:
                results['cycle_count'] += env.cycle_count
                results['idle_time'] += sum(env.idle_times_operator.values())
    
    return (results['cycle_count'] + results['idle_time']) /episodes
    
def main():
    #hparams = [(low, high),        #per var
    #           (low, high)]
    hparams = [(0, 10), #positive_reward_for_divert
               (0, 10), #wrong_sup_at_goal
               (0, 10), #flooding_reward
               (0, 10), #neg_reward_ia
               (0, 10), #negative_reward_for_empty_queue
               (0, 10)] #negative_reward_for_cycle
    
    #define path for the results
    hyperdive_results = join(path, 'rl', 'hyper_parameter', env_name)
    
    #make folder if not exist
    try:
        os.mkdir(hyperdive_results)
    except:
        pass
    
    #run the hyper drive optimization
    hyperdrive(objective=objective,
               hyperparameters=hparams,
               results_path=hyperdive_results,
               checkpoints_path=hyperdive_results,
               model="GP",
               n_iterations=50,
               verbose=True,
               random_state=42)

if __name__=='__main__':
     main()