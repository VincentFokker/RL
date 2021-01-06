## EXTRA REQUIREMENTS
# - psutil
# - plotly
# - requests

import pathlib
import argparse
from os import listdir
from os.path import isfile, join
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import logging
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
"""
Usage of this image generator:
    python make_combined_plots.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -n [RUNNR]
    e.g.
    python make_combined_plots.py -e TestEnv -s Test123 -n 0

"""
#CHANGE LOGGING SETTINGS HERE: #INFO; showing all print statements #DEBUG: show extra info
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment')
    parser.add_argument('-s', '--subdir', type=str, help='Subdir to combine and analyze')
    parser.add_argument('-n', '--name', type=str, help='Which number you want to combine')
    parser.add_argument('-g', '--graph', type=str, help='Which scalar you want to plot')
    args = parser.parse_args()
    path = pathlib.Path().absolute()
    path_to_trained_file = join(path, 'rl', 'trained_models', args.environment, args.subdir)
    items_to_process = [file for file in listdir(path_to_trained_file) if file.startswith(args.name)]
    print('path is: {}'.format(path_to_trained_file))

    if items_to_process == []:
        logging.info('No items found with prefix: {}'.format(args.name))
    else:
        logging.info('Importing and combining {} log files..'.format(len(items_to_process)-1))
        df_reward = pd.DataFrame()
        df_loss = pd.DataFrame()
        for item in items_to_process[1:]:
            focus_path = join(path_to_trained_file, item)
            in_dir = [item for item in listdir(focus_path) if item.startswith('events.out.tfevents')]

            full_path = join(focus_path, in_dir[0])
            logging.debug(full_path)
            logging.info('start import {} / {} from tensorboard event'.format(items_to_process.index(item),len(items_to_process)-1))
            ea = event_accumulator.EventAccumulator(full_path)
            ea.Reload()  # loads events from file
            ea.Tags()
            logging.info('import {} / {} from tensorboard event done'.format(items_to_process.index(item),len(items_to_process)-1))
            tb_rew = pd.DataFrame(ea.Scalars('episode_reward'))
            tb_loss = pd.DataFrame(ea.Scalars('loss/loss'))
            df_reward = df_reward.append(tb_rew)
            df_loss = df_loss.append(tb_loss)
        logging.debug(df_reward.head())
        logging.debug(df_loss.head())
        logging.debug('len total df is {}'.format(len(df_reward)))
        logging.info('Building plots for episode reward and loss..')

        fig = make_subplots(rows=2, cols=1)
        fig.append_trace(go.Scatter(
            x=df_reward['wall_time'],
            y=df_reward['value'],
            name='Episode Reward'
         ), row=1, col=1)

        fig.append_trace(go.Scatter(
            x=df_loss['wall_time'],
            y=df_loss['value'],
            name='Episode loss'
         ), row=2, col=1)

        fig.update_layout(height=1000, width=1800, title_text="Episode reward and loss for run: /{}/{}/{}".format(args.environment, args.subdir, args))
        fig.show()
        logging.info('- Plot is opened in a new browser window -')

        #make matplotlib plot and store
        plt.plot(df_reward['wall_time'], df_reward['value'])
        plt.savefig("images/{}-{}-reward.png".format(args.environment, args.subdir))
        plt.plot(df_loss['wall_time'], df_loss['value'])
        plt.savefig("images/{}-{}-loss.png".format(args.environment, args.subdir))
        logging.info('figure is stored in: /RL/images/')
