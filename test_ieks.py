import gym
import yaml
from rl.environments.SimpleConveyor10 import simple_conveyor_10
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback
from copy import copy

config_path = 'rl/config/0. Old Files/SimpleConveyor10.yml'

with open(config_path, 'r') as f:
    config = yaml.load(f)

model_config = config['models']['PPO2']
n_steps = config['main']['n_steps']
save_every = config['main']['save_every'
]
env = simple_conveyor_10(config)
# multiprocess environment
env_8 = make_vec_env(lambda: env, n_envs=8)


eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)
try:
    try:
        model = PPO2('MlpPolicy', env=env_8, tensorboard_log="./logs/PPO2_tensorboard_log/", **model_config)
        model = PPO2.load("ppo2_SC")
        print("model loaded")
    except:
        model = PPO2('MlpPolicy', env=env_8, tensorboard_log="./logs/PPO2_tensorboard_log/",  **model_config)
        print('new model created')
    model.learn(total_timesteps=15000000)
    model.save("ppo2_SC")
except KeyboardInterrupt:
    print('Saving model..')
    model.save("ppo2_SC")
    print('Done.')