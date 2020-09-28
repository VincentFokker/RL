import argparse
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
"""
Script for plotting results of the logfiles with plotly.

Usage:  python make_plots.py -l [LOGFILE] -t 0
    OR
        python make_plots.py -l [LOGFILE] -t 1 -s [SCALAR]
    WHERE
        SCALARS ARE:
        'loss/entropy_loss','loss/policy_gradient_loss','loss/value_function_loss','loss/approximate_kullback-leibler',
        'loss/clip_factor','loss/loss','input_info/discounted_rewards','input_info/learning_rate','input_info/advantage',
        'input_info/clip_range','input_info/clip_range_vf','input_info/old_neglog_action_probabilty','input_info/old_value_pred',
        'steps','amount_of_items_on_conv','amount_of_items_in_sys','remaining_demand','amount_of_orders_processed','positive_reward',
        'cycle_count','run_count','episode_reward'
"""

def plot_tensorflow_log(path):
    '''builds set of plots from a logfile'''
    # Load the TB data
    # files are large (200+ Mb, might take a while)
    print('start import from tensorboard event')
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()  # loads events from file
    ea.Tags()
    print('import from tensorboard event done')

    Scalars = ea.Tags()['scalars']
    cols = 4
    rows = len(Scalars) // cols + len(Scalars) % cols
    fig = make_subplots(rows=rows, cols=cols)
    print('Building Plots.. ')
    for item in Scalars:
        i = Scalars.index(item) + 1
        col1 = (i - 1) % cols + 1
        row1 = (i + cols - 1) // cols
        fig.append_trace(go.Scatter(
            x=pd.DataFrame(ea.Scalars(item))['wall_time'],
            y=pd.DataFrame(ea.Scalars(item))['value'],
            name=item
        ), row=row1, col=col1)

    fig.update_layout(height=360*rows, width=450*cols, title_text="Episode_reward Subplots")
    fig.show()


def plot_tensorflow_log_specific(path, scalar):
    '''builds set of plots from a logfile'''
    # Load the TB data
    # files are large (200+ Mb, might take a while)
    print('start import from tensorboard event')
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()  # loads events from file
    ea.Tags()
    print('import from tensorboard event done')

    fig = go.Figure(go.Scatter(
        x=pd.DataFrame(ea.Scalars(scalar))['wall_time'],
        y=pd.DataFrame(ea.Scalars(scalar))['value'],
        name=scalar
    ))
    fig.update_layout(height=900  , width=1800, title_text=scalar)
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logdir', type=str, help='Logdir of the file to plot')
    parser.add_argument('-t', '--type', type=int, help='how to plot 0 = all, 1 = specific plot scalar')
    parser.add_argument('-s', '--scalar', type=str, help='Variable of the scalar that needs to be plotted')
    args = parser.parse_args()

    log_file = args.logdir
    scalars = ['loss/entropy_loss',
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
 'amount_of_items_on_conv',
 'amount_of_items_in_sys',
 'remaining_demand',
 'amount_of_orders_processed',
 'positive_reward',
 'cycle_count',
 'run_count',
 'episode_reward']

    if args.type == 0:
        plot_tensorflow_log(log_file)
    elif args.type == 1:
        if args.scalar in scalars:
            plot_tensorflow_log_specific(log_file, args.scalar)
        else:
            print('Scalar not in set of possible scalars!')

    else:
        print('Type not allowed! select 0 for all plots, select 1 for specific plot + -s SCALARNAME')
