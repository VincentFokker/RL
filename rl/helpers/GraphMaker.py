import argparse
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pathlib
from os import listdir, walk
from os.path import join, isfile
import matplotlib.pyplot as plt

##Used functions
def smooth(scalars, weight):
    """
    Smoothes the datapoints
    :param scalars: list with datapoints
    :param weight: float between 0 and 1, how smooth
    :return: the smoothed data points
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed

def Load_data(environment, subdir, scalar='episode_reward', smoothing=0.96):
    """
    Load The data from the different logs, combine, return dataframe
    input:
    - ENVIRONMENT
    - SUBDIR
    (-RUNNR)
    Scalars to plot (alternatively):
    
    'loss/entropy_loss',
     'loss/policy_gradient_loss',
     'loss/value_function_loss',
     'loss/approximate_kullback-leibler',
     'loss/clip_factor',
     'loss/loss',
     'input_info/discounted_rewards',
     'input_info/learning_rate',
     'input_info/advantage',
     'input_info/clip_range',
     'input_info/clip_range_vf',
     'input_info/old_neglog_action_probabilty',
     'input_info/old_value_pred',
     'steps',
     'episode_reward'
     
    """
    path = pathlib.Path().absolute()
    specified_path = join(path, 'rl', 'trained_models', environment, subdir)
    subfolders = [x[0] for x in walk(specified_path)][1:]
    logfiles = [join(folder, listdir(folder)[0]) for folder in subfolders]

    df = []
    for logpath in logfiles:
        print('start import {} / {} from tensorboard event'.format(logfiles.index(logpath), len(logfiles)), end='\r')
        ea = event_accumulator.EventAccumulator(logpath)
        ea.Reload()  # loads events from file
        print('import from tensorboard event done')
        df.append(pd.DataFrame(ea.Scalars(scalar)))

    for i in range(1,len(df)):
        df[i].step = df[i].step + df[i-1].step.max()

    df_out = pd.DataFrame()
    for df_s in df:
        df_out = df_out.append(df_s)

    df_out
        
    df_out['value_s'] = smooth(df_out.value.to_list(), smoothing)
    return df_out

def plotly_data(df):
    """
    Takes input df from the Load_data Function, applies smoothing and returns plot"""
    
    fig = px.line(df, x= 'wall_time', y=['value', 'value_s'], color_discrete_map={
                 "value": "#f9d1b3",
                 "value_s": "#f06d07"
             }, labels=['iets', 'niets'])
    fig.show()

def matplot_data(df, scalar, timevar='wall_time'):
    """builds matplotlib figure of the data"""
    plt.style.use('ggplot')
    plt.figure(figsize=(15, 5))
    plt.plot(df[timevar], df.value, color ='#f9d1b3')
    plt.plot(df[timevar], df.value_s, color = '#f06d07')
    plt.xlabel('Timestep')
    plt.ylabel(scalar)
    #plt.grid(color='grey')
    #plt.axhline(color='black')
    plt.show()