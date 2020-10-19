from rl.environments.SimpleConveyor10 import SimpleConveyor10
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import cv2
import yaml
import argparse
from rl.baselines import get_parameters
import logging

"""
Usage of this tester:
    python test3.py -e [ENVIRONMENT_NAME] -n [NUMEPS] -m [MODELPATH]
    e.g.
    python test3.py -e TestEnv -s Test123 -n 0

"""
#CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements #DEBUG: show extra info
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment')
    parser.add_argument('-n', '--numbereps', type=int, help='Number of episodes to run')
    parser.add_argument('-m', '--modelpath', type=str, help='Path to the specific model to check')
    args = parser.parse_args()

config_path = 'rl/config/SimpleConveyor10.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f)
#config = get_parameters(args.environment)
env = SimpleConveyor10(config)

modelpath = args.modelpath
model = PPO2.load(modelpath, env=DummyVecEnv([lambda: env]))

for episode in range(args.numbereps):
    # Run an episode
    state = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)

        # monitor figure
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(state.reshape(6, 6))
        ax1.set_title('State')
        ax2.bar(x=[0, 1, 2, 3], height=model.action_probability(state))
        ax2.set_title('Action Distribution')
        f.savefig('buffer_img.png')
        plt.close(f)
        image = cv2.imread('buffer_img.png')
        cv2.imshow("image", image)
        cv2.waitKey(1)
        state, reward, done, _ = env.step(action)
        env.render()