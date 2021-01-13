from rl.helpers import Load_data
import pathlib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
path = pathlib.Path().absolute()
scalars = ['episode_reward', 'loss/loss']
timevar = 'step'  # wall_time or step

to_combine=[
    ['ConveyorEnv12','20210113_0000'], #2x2
    ['ConveyorEnv12','20210113_0100'], #2x3
    ['ConveyorEnv12','20210113_0200'], #2x3
    ['ConveyorEnv12','20210113_0300'], #3x3
    ['ConveyorEnv12','20210113_0400'], #4x3
    ['ConveyorEnv12','20210113_0500'] #5x3
]

if __name__ == "__main__":
    #parse the arguments from the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--term', type=str, help='Term to identify specific plot')
    args = parser.parse_args()

    all_rewardplots = []
    for env, subdir in to_combine:
        data = Load_data(env, subdir, scalar='episode_reward')
        all_rewardplots.append(data)

    plt.style.use('ggplot')

    timevar = 'step'     #step or wall_time
    labels = ['1x2 design', '2x2 design', '2x3 design', '3x3 design', '4x3 design', '5x3 design']

    ### FOR REWARD
    #determine colors
    intdata = (np.array(sns.color_palette("viridis", len(all_rewardplots))) * 255).astype(int)
    hexen = ['#%02x%02x%02x' % tuple(code) for code in intdata]

    fig = plt.figure(figsize=(15, 6))
    for idx, df in enumerate(all_rewardplots):
        plt.plot(df[timevar], df.value, color =hexen[idx], alpha=0.10)
    for idx, df in enumerate(all_rewardplots):
        plt.plot(df[timevar], df.value_s, color = hexen[idx], label=labels[idx])
    plt.xlabel('Timestep')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.savefig('figures/episode_reward_designs_{}.png'.format(args.term))
    print('Combined plot stored in /figures/')


    for idx, df in enumerate(all_rewardplots):
        fig = plt.figure(figsize=(15, 6))
        plt.plot(df[timevar], df.value, color =hexen[idx], alpha=0.10)
        plt.plot(df[timevar], df.value_s, color = hexen[idx], label=labels[idx])
        plt.xlabel('Timestep')
        plt.ylabel('Episode Reward')
        plt.legend()
        plt.savefig('figures/reward_{}_{}.png'.format(labels[idx], args.term))

    ### FOR LOSS
    all_loss = []
    for env, subdir in to_combine:
        data = Load_data(env, subdir, scalar='loss/loss',smoothing=0.99)
        all_loss.append(data)

    # determine colors
    intdata = (np.array(sns.color_palette("viridis", len(all_loss)+2)) * 255).astype(int)
    hexen = ['#%02x%02x%02x' % tuple(code) for code in intdata[2:]]

    ## combined plot
    fig = plt.figure(figsize=(15, 5))#, dpi=300)
    for idx, df in enumerate(all_loss):
        plt.plot(df[timevar], df.value, color =hexen[idx], alpha=0.10)
    for idx, df in enumerate(all_loss):
        plt.plot(df[timevar], df.value_s, color = hexen[idx], label=labels[idx])
    plt.xlabel('Timestep')
    plt.ylabel('Loss / loss')
    plt.legend()
    plt.savefig('figures/loss_designs_{}.png'.format(args.term))

    ## individual plots
    for idx, df in enumerate(all_loss):
        fig = plt.figure(figsize=(15, 6))
        plt.plot(df[timevar], df.value, color =hexen[idx], alpha=0.15)
        plt.plot(df[timevar], df.value_s, color = hexen[idx], label=labels[idx])
        plt.xlabel('Timestep')
        plt.ylabel('Loss/loss')
        plt.legend()
        plt.savefig('figures/loss_{}_{}.png'.format(labels[idx], args.term))

    print('All plots generated, can be found in: /Figures/')