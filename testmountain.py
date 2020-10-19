import gym

from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.evaluation import evaluate_policy

# Separate evaluation env
eval_env = gym.make('Pendulum-v0')
# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)

model = PPO2('MlpPolicy', eval_env)
model.learn(5000, callback=eval_callback)
model.save('testsav')

del model

model = PPO2.load('testsav')
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print('Mean Reward : {}, Standardized Reward : {}'.format(mean_reward, std_reward))