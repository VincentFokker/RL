import rl.environments
import yaml, os
import argparse
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import rl.helpers
import gym
from stable_baselines.gail import generate_expert_traj
from stable_baselines.gail import ExpertDataset
from stable_baselines import PPO2
import pathlib
from os.path import join
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import EvalCallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment.')
    parser.add_argument('-s', '--subdir', type=str, help='Subdir to combine and analyze.')
    parser.add_argument('-n', '--numepisodes', type=int, help='Number of episodes to generate.')
    args = parser.parse_args()
    path = pathlib.Path().absolute()
    specified_path = join(path, 'rl', 'trained_models', args.environment, args.subdir)

    config_path = 'rl/config/{}.yml'.format(args.environment)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env_obj = getattr(rl.environments, args.environment)
    env = env_obj(config)


    def decode_binary(binary_array):
        return [int("".join([str(n) for n in [int(l) for l in list(binary_array[i - 2:i])]]), 2) for i in
                range(2, len(binary_array) + 2, 2)]


    def decode_action(order_type, goal):
        return (order_type - 1) * env.amount_of_gtps + goal


    def dummy_expert(obs):
        """
        Based on observation , heuristic determines the policy  ( can only take observation [4, 15, 16, 17, 18])

        :param _obs: (np.ndarray) Current observation
        :return: (np.ndarray) action taken by the expert
        """
        threshold = 15

        demands = decode_binary(obs[:2 * env.amount_of_gtps * env.in_que_observed])
        queue_demands = [demands[i * env.in_que_observed: env.in_que_observed + i * env.in_que_observed] for i in
                         range(env.amount_of_gtps)]
        W_rpt = obs[
                2 * env.amount_of_gtps * env.in_que_observed:2 * env.amount_of_gtps * env.in_que_observed + env.amount_of_gtps]
        max_time_w = 6 if env.amount_of_outputs == 1 else 30 if env.amount_of_outputs == 2 else 60
        W_rpt = W_rpt * max_time_w
        Q_rpt = obs[
                2 * env.amount_of_gtps * env.in_que_observed + env.amount_of_gtps:2 * env.amount_of_gtps * env.in_que_observed + env.amount_of_gtps * 2]
        max_time_q = max_time_w * env.gtp_buffer_length
        Q_rpt = Q_rpt * max_time_q
        P_rpt = obs[
                2 * env.amount_of_gtps * env.in_que_observed + env.amount_of_gtps * 2:2 * env.amount_of_gtps * env.in_que_observed + env.amount_of_gtps * 3]
        P_rpt = P_rpt * env.pipeline_length
        in_pipe = obs[2 * env.amount_of_gtps * env.in_que_observed + env.amount_of_gtps * 3:2 * env.amount_of_gtps * env.in_que_observed + env.amount_of_gtps * 4]
        #in_pipe = obs[-env.amount_of_gtps:]
        in_pipe = in_pipe * env.pipeline_length
        in_pipe = in_pipe.astype(int)

        actions_list = []
        for workstation in range(env.amount_of_gtps)[::-1]:
            total_rpt = W_rpt[workstation] + Q_rpt[workstation] + P_rpt[workstation]
            total_pipe = env.pipeline_length + env.gtp_buffer_length + workstation * 4 + 2

            if total_rpt - total_pipe < threshold:
                try:
                    current_demand = queue_demands[workstation][in_pipe[workstation]]
                    actions_list.append((current_demand, workstation + 1))
                except:
                    pass

        try:
            order_type, goal = actions_list[0]
            actions_list = actions_list[1:]

        except:
            order_type, goal = 0, 0

        if order_type == 0 and goal == 0:
            action = 0

        else:
            action = decode_action(order_type, goal)

        return action
        # make folder if not exist
    try:
        os.mkdir(specified_path)
    except:
        pass

    ## Generate Data based on heuristic for pre-training
    # Data will be saved in a numpy archive named `heuristic_expert.npz`
    env.reset()
    generate_expert_traj(dummy_expert, join(specified_path,'heuristic_expert'), env, n_episodes=args.numepisodes)