from rl.helpers import Load_data, matplot_data
import pathlib
import argparse
from os.path import join

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment.')
    parser.add_argument('-s', '--subdir', type=str, help='Subdir to combine and analyze.')
    args = parser.parse_args()
    path = pathlib.Path().absolute()
    specified_path = join(path, 'rl', 'trained_models', args.environment, args.subdir)

    scalars = ['episode_reward', 'loss/loss']
    timevar = 'step'  # wall_time or step

    #save reward figure
    reward = Load_data(args.environment, args.subdir, scalar='episode_reward')
    figure = matplot_data(reward, 'episode_reward', timevar=timevar, show=False)
    figure.savefig(join(specified_path, 'episode_reward.png'))

    #save loss figure
    loss = Load_data(args.environment, args.subdir, scalar='loss/loss')
    figure = matplot_data(loss, 'loss/loss', timevar=timevar, show=False)
    figure.savefig(join(specified_path, 'loss_loss.png'))