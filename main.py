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
from env.wrappers import rdkit_reward_wrapper

from tqc import DEVICE
from tqc.trainer import Trainer
from tqc.actor_critic import Actor, Critic
from tqc.replay_buffer import ReplayBuffer
from tqc.utils import run_policy_eval_and_explore, eval_policy_multiple_timelimits
from tqc.utils import ActionScaleScheduler, TIMELIMITS


class Logger:
    def __init__(self, experiment_folder, config):
        if os.path.exists(experiment_folder):
            raise Exception('Experiment folder exists, apparent seed conflict!')
        os.makedirs(experiment_folder)
        self.metrics_file = experiment_folder / "metrics.json"
        self.metrics_file.touch()
        with open(experiment_folder / 'config.json', 'w') as config_file:
            json.dump(config.__dict__, config_file)

        self._keep_n_episodes = 10
        self.exploration_episode_lengths = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_returns = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_mean_Q = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_final_energy = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_final_rl_energy = deque(maxlen=self._keep_n_episodes)
        self.exploration_not_converged = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_number = 0

    def log(self, metrics):
        metrics['Exploration episodes number'] = self.exploration_episode_number
        for name, d in zip(
                ['episode length',
                 'episode return',
                 'episode mean Q',
                 'episode final energy',
                 'episode final rl energy',
                 'episode not converged'],
                [self.exploration_episode_lengths,
                 self.exploration_episode_returns,
                 self.exploration_episode_mean_Q,
                 self.exploration_episode_final_energy,
                 self.exploration_episode_final_rl_energy,
                 self.exploration_not_converged]
            ):
            metrics[f'Exploration {name}, mean'] = np.mean(d)
            metrics[f'Exploration {name}, std'] = np.std(d)
        with open(self.metrics_file, 'a') as out_metrics:
            json.dump(metrics, out_metrics)
            out_metrics.write('\n')

    def update_evaluation_statistics(self,
                                     episode_length,
                                     episode_return,
                                     episode_mean_Q,
                                     episode_final_energy,
                                     episode_final_rl_energy,
                                     not_converged):
        self.exploration_episode_number += 1
        self.exploration_episode_lengths.append(episode_length)
        self.exploration_episode_returns.append(episode_return)
        self.exploration_episode_mean_Q.append(episode_mean_Q)
        self.exploration_episode_final_energy.append(episode_final_energy)
        self.exploration_episode_final_rl_energy.append(episode_final_rl_energy)
        self.exploration_not_converged.append(not_converged)


def main(args, experiment_folder):
    # Tmp set env name
    args.env = args.db_path.split('/')[-1].split('.')[0]
    print(args.env)
    # Initialize logger
    logger = Logger(experiment_folder, args)
    
    # Initialize env
    # For a debugging experiment with single initial_conformation
    if args.sample_initial_conformation:
        init_conf_index = np.random.randint(950000)
    else:
        init_conf_index = None
    env_kwargs = {
        'db_path': args.db_path,
        'timelimit': args.timelimit,
        'done_on_timelimit': args.done_on_timelimit,
        'num_initial_conformations': args.num_initial_conformations,
        'initial_conformation_index': init_conf_index,
        'inject_noise': args.inject_noise,
        'noise_std': args.noise_std,
        'calculate_mean_std': args.calculate_mean_std_energy,
        'remove_hydrogen': args.remove_hydrogen,
    }
    env = env_fn(DEVICE, **env_kwargs)
    
    # Initialize eval envs
    env_kwargs['inject_noise'] = False
    eval_env = env_fn(DEVICE, **env_kwargs)
    # For evaluation on multiple timestamps
    env_kwargs['timelimit'] = max(TIMELIMITS)
    eval_env_long = env_fn(DEVICE, **env_kwargs)
    
    # Seed env
    env.seed(args.seed)
    eval_env.seed(args.seed)
    eval_env_long.seed(args.seed)

    # Initialize reward wrapper for training
    reward_wrapper_kwargs = {
        'env': env,
        'minimize_on_every_step': args.minimize_on_every_step,
        'remove_hydrogen': args.remove_hydrogen,
        'molecules_xyz_prefix': args.molecules_xyz_prefix,
        'M': args.M
    }
    env = rdkit_reward_wrapper(**reward_wrapper_kwargs)
    
    # Initialize reward wrappers for evaluation
    reward_wrapper_kwargs['env'] = eval_env
    eval_env = rdkit_reward_wrapper(**reward_wrapper_kwargs)
    reward_wrapper_kwargs['env'] = eval_env_long
    reward_wrapper_kwargs.update({
        'env': eval_env_long,
        'minimize_on_every_step': True
    })
    eval_env_long = rdkit_reward_wrapper(**reward_wrapper_kwargs)

    # Initialize action_scale scheduler
    action_scale_scheduler = ActionScaleScheduler(action_scale_init=args.action_scale_init, 
                                                  action_scale_end=args.action_scale_end,
                                                  n_step_end=args.action_scale_n_step_end,
                                                  mode=args.action_scale_mode)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(DEVICE, args.replay_buffer_size)

    # Inititalize actor and critic
    schnet_args = {
        'n_interactions': args.n_interactions,
        'cutoff': args.cutoff,
        'n_gaussians': args.n_gaussians,
    }
    # SchNet backbone is shared between actor and all critics
    actor = Actor(schnet_args, args.actor_out_embedding_size, action_scale_scheduler).to(DEVICE)
    critic = Critic(schnet_args, args.n_nets, args.critic_out_embedding_size, args.n_quantiles).to(DEVICE)
    critic_target = copy.deepcopy(critic)

    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets
    # Target entropy must differ for molecules of different sizes.
    # To achieve this we fix per-atom entropy instead of the entropy of the molecule.
    per_atom_target_entropy = 3 * (-1 + np.log([args.target_entropy_action_scale])).item()
    trainer = Trainer(actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      top_quantiles_to_drop=top_quantiles_to_drop,
                      discount=args.discount,
                      tau=args.tau,
                      log_alpha=np.log([args.initial_alpha]).item(),
                      actor_lr=args.actor_lr,
                      critic_lr= args.critic_lr,
                      alpha_lr=args.alpha_lr,
                      per_atom_target_entropy=per_atom_target_entropy)

    state, done = env.reset(), False
    episode_return = 0
    episode_mean_Q = 0
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
        action_scale_scheduler.update(t)
        with torch.no_grad():
            action = actor.select_action(state)
            mean_Q = critic_target(state, torch.FloatTensor(action).unsqueeze(0).to(DEVICE)).mean().item()

        next_state, reward, done, info = env.step(action)
        # Done on every step or at the end of the episode
        done = done or args.greedy
        episode_timesteps += 1
        ep_end = episode_timesteps >= args.timelimit
        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state
        episode_return += reward
        episode_mean_Q += mean_Q

        # Train agent after collecting sufficient data
        if t >= args.batch_size:
            step_metrics = trainer.train(replay_buffer, args.batch_size)
        else:
            step_metrics = dict()
        step_metrics['Timestamp'] = str(datetime.datetime.now())
        step_metrics['Action_scale'] = action_scale_scheduler.get_action_scale()
        step_metrics['Action_norm'] = np.linalg.norm(action, axis=1).mean().item()

        if done or (not args.greedy and ep_end):
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            if not 'final_energy' in info:
                info['final_energy'] = 10.0
            if not 'final_rl_energy' in info:
                info['final_rl_energy'] = 10.0
            if not 'not_converged' in info:
                info['not_converged'] = 1.0

            logger.update_evaluation_statistics(episode_timesteps,
                                                episode_return,
                                                episode_mean_Q,
                                                info['final_energy'],
                                                info['final_rl_energy'],
                                                info['not_converged'])

            episode_return = 0
            episode_mean_Q = 0
            episode_num += 1

        if ep_end:
            episode_timesteps = 0
            # Reset environment
            state, done = env.reset(), False

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            step_metrics['Total_timesteps'] = t + 1
            step_metrics.update(run_policy_eval_and_explore(actor, eval_env, args.timelimit, args.n_eval_runs))
            if args.evaluate_multiple_timelimits:
                step_metrics.update(eval_policy_multiple_timelimits(actor, eval_env_long, args.M, args.n_eval_runs))
            logger.log(step_metrics)

        if t in full_checkpoints and args.save_checkpoints:
            trainer_save_name = f'{experiment_folder}/iter_{t}'
            trainer.save(trainer_save_name)
            #with open(f'{experiment_folder}/iter_{t}_replay', 'wb') as outF:
            #    pickle.dump(replay_buffer, outF)
            # Remove previous checkpoint?
        elif (t + 1) % args.light_checkpoint_freq == 0 and args.save_checkpoints:
            trainer_save_name = f'{experiment_folder}/iter_{t}'
            trainer.light_save(trainer_save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Env args
    parser.add_argument("--db_path", default="env/data/malonaldehyde.db", type=str, help="Path to molecules database")
    parser.add_argument("--num_initial_conformations", default=50000, type=int, help="Number of initial molecule conformations to sample from DB")
    parser.add_argument("--sample_initial_conformation", default=False, type=bool, help="Sample new conformation for every seed")
    parser.add_argument("--timelimit", default=100, type=int, help="Timelimit for MD env")
    parser.add_argument("--greedy", default=False, type=bool, help="Returns done on every step independent of the timelimit")
    parser.add_argument("--done_on_timelimit", type=bool, default=False, help="Env returns done when timelimit is reached")
    parser.add_argument("--inject_noise", type=bool, default=False, help="Whether to inject random noise into initial states")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Std of the injected noise")
    parser.add_argument("--calculate_mean_std_energy", type=bool, default=False, help="Calculate mean, std of energy of database")
    parser.add_argument("--remove_hydrogen", type=bool, default=False, help="Whether to remove hydrogen atoms from the molecule")
    # Reward args
    parser.add_argument("--minimize_on_every_step", type=bool, default=False, help="Whether to minimize conformation with rdkit on every step")
    parser.add_argument("--M", type=int, default=10, help="Number of steps to run rdkit minimization for")
    parser.add_argument("--molecules_xyz_prefix", type=str, default="", help="Path to env/ folder. For cluster compatability")
    # Action scale args. Action scale bounds actions to [-action_scale, action_scale]
    parser.add_argument("--action_scale_init", default=0.01, type=float, help="Initial value of action_scale")
    parser.add_argument("--action_scale_end", default=0.05, type=float, help="Final value of action_scale")
    parser.add_argument("--action_scale_n_step_end", default=int(8e5), type=int, help="Step at which the final value of action_scale is reached")
    parser.add_argument("--action_scale_mode", choices=["constant", "discrete", "continuous"], default="constant", help="Mode of action scale scheduler")
    parser.add_argument("--target_entropy_action_scale", default=0.01, type=float, help="Controls target entropy of the distribution")
    # Schnet args
    parser.add_argument("--n_interactions", default=3, type=int, help="Number of interaction blocks for Schnet in actor/critic")
    parser.add_argument("--cutoff", default=20.0, type=float, help="Cutoff for Schnet in actor/critic")
    parser.add_argument("--n_gaussians", default=50, type=int, help="Number of Gaussians for Schnet in actor/critic")
    # Actor args
    parser.add_argument("--actor_out_embedding_size", default=128, type=int, help="Output embedding size for actor")
    # Critic args
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument("--critic_out_embedding_size", default=128, type=int, help="Critic out embedding size")
    # Eval args
    parser.add_argument("--eval_freq", default=1e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--evaluate_multiple_timelimits", default=False, type=bool, help="Evaluate policy at multiple timelimits")
    parser.add_argument("--n_eval_runs", default=10, type=int, help="Number of evaluation episodes")
    # Trainer args
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--replay_buffer_size", default=int(2e5), type=int, help="Size of replay buffer")
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--actor_lr", default=3e-4, type=float, help="Actor learning rate")
    parser.add_argument("--critic_lr", default=3e-4, type=float, help="Critic learning rate")
    parser.add_argument("--alpha_lr", default=3e-4, type=float, help="Alpha learning rate")
    parser.add_argument("--initial_alpha", default=1.0, type=float, help="Initial value for alpha")
    # Other args
    parser.add_argument("--exp_name", required=True, type=str, help="Name of the experiment")
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--light_checkpoint_freq", type=int, default=200000)
    parser.add_argument("--save_checkpoints", type=bool, default=False, help="Save light and full checkpoints")
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