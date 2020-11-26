from rl.environments import *
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import cv2
import yaml, pathlib
from os.path import join
import argparse
from rl.baselines import get_parameters
import logging
import numpy as np
import rl

"""
Usage of this tester:
    python test3.py -e [ENVIRONMENT_NAME] -n [NUMEPS] -m [MODELPATH]
    e.g.
    python test3.py -e TestEnv -n 1 -m /path/to/file.zip

"""
#CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements #DEBUG: show extra info
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment')
    parser.add_argument('-s', '--subdir', type=str, help='Subdir to look at')
    parser.add_argument('-n', '--name', type=str, help='Name of the model to run')
    parser.add_argument('-w', '--waitkey', type=int, help='Waitkey for the render')
    parser.add_argument('-r', '--render', action='store_true', help='Render the agents.')
    parser.add_argument('-p', '--probability', action='store_true', help='Renders action probability.')
    parser.add_argument('-d', '--deterministic', action='store_true', help='If the agent should be deterministic.')
    args = parser.parse_args()

    path = pathlib.Path().absolute()
    specified_path = join(path, 'rl', 'trained_models', args.environment, args.subdir)

    with open(join(specified_path, 'config.yml'), 'r') as f:
        config = yaml.safe_load(f)

    config_env = config['environment']
    amount_output = config_env['amount_of_outputs']
    amount_gtps = config_env['amount_of_gtps']
    stop = amount_gtps * amount_output + 1
    #load environment with config variables
    env_obj = getattr(rl.environments, args.environment)
    env = env_obj(config)

    modelpath = join(specified_path, 'best_model.zip')
    model = PPO2.load(modelpath, env=DummyVecEnv([lambda: env]))
    print(args.render)
    for episode in range(10):
        # Run an episode
        state = env.reset()
        size = state.shape[0]
        done = False
        meta_data = []
        while not done:
            action, _ = model.predict(state, deterministic=args.deterministic)
            logging.debug(model.action_probability(state))

            if args.probability:
                ## monitoring
                r = 4
                state_n = state
                for i in range(r):
                    state_n = np.append(state_n, state_n)
                state_n = state_n.reshape(2 ** r, size)

                # monitor figure
                f, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(state_n)
                ax1.set_title('State')
                ax2.bar(x=range(0,amount_gtps*amount_output+1), height=model.action_probability(state))
                ax2.set_title('Action Distribution')
                f.savefig('buffer_img.png')
                plt.close(f)
                image = cv2.imread('buffer_img.png')
                cv2.imshow("image", image)
                cv2.waitKey(args.waitkey)

            state, reward, done, tc = env.step(action)
            meta_data.append(tc)
            if args.render:
                env.render()

        #write the meta_data to file
        with open(join(specified_path,'performance_metrics.txt'), 'a') as f:
            f.write("%s\n" % meta_data[-1])