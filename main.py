import argparse
import copy
import datetime
import json
import numpy as np
import torch
import os
import pickle
import random
from pathlib import Path
from collections import deque


from env.moldynamics_env import env_fn
from env.wrappers import schnet_reward_wrapper, rdkit_reward_wrapper

from tqc import DEVICE
from tqc.trainer import Trainer
from tqc.actor_critic import Actor, Critic
from tqc.replay_buffer import ReplayBuffer
from tqc.functions import eval_policy


EPISODE_LENGTH = 100

class Logger:
    def __init__(self, experiment_folder, config):
        if os.path.exists(experiment_folder):
            raise Exception('Experiment folder exists, apparent seed conflict!')
        os.makedirs(experiment_folder)
        self.metrics_file = experiment_folder / "metrics.json"
        self.metrics_file.touch()
        with open(experiment_folder / 'config.json', 'w') as config_file:
            json.dump(config.__dict__, config_file)

        self._keep_n_episodes = 5
        self.exploration_episode_lengths = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_returns = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_info_returns = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_number = 0

    def log(self, metrics):
        metrics['Exploration episodes number'] = self.exploration_episode_number
        for name, d in zip(['episode length', 'episode return'], [self.exploration_episode_lengths, self.exploration_episode_returns]):
            metrics[f'Exploration {name}, mean'] = np.mean(d)
            metrics[f'Exploration {name}, std'] = np.std(d)
        with open(self.metrics_file, 'a') as out_metrics:
            json.dump(metrics, out_metrics)
            out_metrics.write('\n')

    def update_evaluation_statistics(self, episode_length, episode_return, episode_info_return):
        self.exploration_episode_number += 1
        self.exploration_episode_lengths.append(episode_length)
        self.exploration_episode_returns.append(episode_return)
        self.exploration_episode_info_returns.append(episode_info_return)


def main(args, experiment_folder):
    # --- Init ---
    # Tmp set env name
    args.env = "Malonaldehyde"
    logger = Logger(experiment_folder, args)
    # Initialize env
    trajectory_dir = experiment_folder / 'trajectory'
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)
    env = env_fn(DEVICE, multiagent=False, db_path=args.db_path, timelimit=args.timelimit,
                 calculate_mean_std=args.calculate_mean_std_energy, exp_folder=trajectory_dir)
    eval_env = env_fn(DEVICE, multiagent=False, db_path=args.db_path, timelimit=args.timelimit,
                      calculate_mean_std=args.calculate_mean_std_energy, exp_folder=trajectory_dir)

    # Initialize reward wrapper
    if args.reward == 'schnet':
        env = schnet_reward_wrapper(env, multiagent=False, schnet_model_path=args.schnet_model_path,
                                    reward_delta=args.reward_delta, device=DEVICE)
        eval_env = env = schnet_reward_wrapper(env, multiagent=False, schnet_model_path=args.schnet_model_path,
                                               reward_delta=args.reward_delta, device=DEVICE)
    elif args.reward == 'rdkit':
        env = rdkit_reward_wrapper(env, multiagent=False, molecule_path=args.molecule_path)
        eval_env = rdkit_reward_wrapper(env, multiagent=False, molecule_path=args.molecule_path)
    elif args.reward == 'both':
        env = rdkit_reward_wrapper(env, multiagent=False, molecule_path=args.molecule_path)
        env = schnet_reward_wrapper(env, multiagent=False, schnet_model_path=args.schnet_model_path,
                                    reward_delta=args.reward_delta, device=DEVICE)
        eval_env = rdkit_reward_wrapper(env, multiagent=False, molecule_path=args.molecule_path)
        eval_env = schnet_reward_wrapper(env, multiagent=False, schnet_model_path=args.schnet_model_path,
                                         reward_delta=args.reward_delta,  device=DEVICE)

    state_dict_names, \
    state_dims, \
    state_dict_dtypes = (zip(*[(k, box.shape, box.dtype) for k, box in env.observation_space.items()]))
    action_dim = env.action_space.shape
    schnet_args = {
        'n_interactions': args.n_interactions,
        'cutoff': args.cutoff,
        'n_gaussians': args.n_gaussians,
    }

    replay_buffer = ReplayBuffer(state_dict_names, state_dict_dtypes, state_dims, action_dim, DEVICE)
    actor = Actor(schnet_args, out_embedding_size=args.actor_out_embedding_size, action_scale=args.action_scale).to(DEVICE)
    critic = Critic(schnet_args, args.n_nets, args.n_quantiles).to(DEVICE)
    critic_target = copy.deepcopy(critic)

    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets

    trainer = Trainer(actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      top_quantiles_to_drop=top_quantiles_to_drop,
                      discount=args.discount,
                      tau=args.tau,
                      target_entropy=-np.prod(env.action_space.shape).item())

    state, done = env.reset(), False
    episode_return = 0
    episode_info_return = 0
    episode_timesteps = 0
    episode_num = 0

    actor.train()
    full_checkpoints = [int(args.max_timesteps / 3), int(args.max_timesteps * 2 / 3), int(args.max_timesteps) - 1]
    if args.load_model is not None:
        start_iter = int(args.load_model.split('/')[-1].split('_')[1]) + 1
        trainer.load(args.load_model)
        replay_buffer = pickle.load(open(f'{args.load_model}_replay', 'rb'))
    else:
        start_iter = 0
    for t in range(start_iter, int(args.max_timesteps)):
        with torch.no_grad():
            action = actor.select_action(state)

        next_state, reward, done, info = env.step(action)
        episode_timesteps += 1
        ep_end = done or episode_timesteps >= EPISODE_LENGTH
        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state
        episode_return += reward
        if 'rdkit_reward' in info:
            episode_info_return += info['rdkit_reward']

        # Train agent after collecting sufficient data
        if t >= args.batch_size:
            step_metrics = trainer.train(replay_buffer, args.batch_size)
        else:
            step_metrics = dict()
        step_metrics['Timestamp'] = str(datetime.datetime.now())

        if ep_end:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            logger.update_evaluation_statistics(episode_timesteps, episode_return, episode_info_return)
            # Reset environment
            state, done = env.reset(), False

            episode_return = 0
            episode_info_return = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            step_metrics['Total_timesteps'] = t + 1
            step_metrics['Evaluation_returns'], step_metrics['RDKit_evaluation_returns'] = eval_policy(actor, eval_env, EPISODE_LENGTH)
            logger.log(step_metrics)

        if t in full_checkpoints:
            trainer_save_name = f'{experiment_folder}/iter_{t}'
            trainer.save(trainer_save_name)
            with open(f'{experiment_folder}/iter_{t}_replay', 'wb') as outF:
                pickle.dump(replay_buffer, outF)
            # Remove previous checkpoint?
        elif (t + 1) % args.light_checkpoint_freq == 0:
            trainer_save_name = f'{experiment_folder}/iter_{t}'
            trainer.light_save(trainer_save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Env args
    parser.add_argument("--db_path", default="env/data/malonaldehyde.db", type=str, help="Path to molecules database")
    parser.add_argument("--timelimit", default=100, type=int, help="Timelimit for MD env")
    parser.add_argument("--schnet_model_path", default="env/schnet_model/schnet_model_3_blocks", type=str, help="Path to trained schnet model")
    parser.add_argument("--molecule_path", default="env/molecules_xyz/malonaldehyde.xyz", type=str, help="Path to example .xyz file")
    parser.add_argument("--calculate_mean_std_energy", action="store_true", default=False, help="Calculate mean, std of energy of database")
    parser.add_argument("--reward", default="both", choices=["schnet", "rdkit", "both"], help="Type of reward for MD env")
    parser.add_argument("--reward_delta", action="store_true", default=False, help="Use delta of energy as reward")
    # Schnet args
    parser.add_argument("--n_interactions", default=3, type=int, help="Number of interaction blocks for Schnet in actor/critic")
    parser.add_argument("--cutoff", default=20.0, type=float, help="Cutoff for Schnet in actor/critic")
    parser.add_argument("--n_gaussians", default=50, type=int, help="Number of Gaussians for Schnet in actor/critic")
    # Actor args
    parser.add_argument("--actor_out_embedding_size", default=128, type=int, help="Output embedding size for actor")
    parser.add_argument("--action_scale", default=0.01, type=float, help="Bounds actions to [-action_scale, action_scale]")
    # Other args
    parser.add_argument("--exp_name", required=True, type=str, help="Name of the experiment")
    parser.add_argument("--eval_freq", default=1e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--light_checkpoint_freq", type=int, default=200000)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--log_dir", default='.')
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    #args.git_sha = get_current_gitsha()

    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    exp_folder = log_dir / f'{args.exp_name}_{start_time}_{args.seed}'

    main(args, exp_folder)