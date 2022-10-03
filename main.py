import argparse
import datetime
import json
import numpy as np
import os
import pickle
import random
import time
import torch

from pathlib import Path
from collections import deque

from env.moldynamics_env import env_fn
from env.wrappers import reward_wrapper
from env.make_envs import make_envs

from rl import DEVICE
from rl.tqc import TQC
from rl.ppo import PPO
from rl.actor_critic_tqc import TQCPolicy
from rl.actor_critic_ppo import PPOPolicy
from rl.replay_buffer import ReplayBufferPPO, ReplayBufferTQC
from rl.utils import eval_policy, ignore_extra_args,\
                     recollate_batch, calculate_action_norm
from rl.utils import ActionScaleScheduler, TimelimitScheduler

policies = {
    "PPO": ignore_extra_args(PPOPolicy),
    "TQC": ignore_extra_args(TQCPolicy),
}

trainers = {
    "PPO": ignore_extra_args(PPO),
    "TQC": ignore_extra_args(TQC)
}

replay_buffers = {
    "PPO": ignore_extra_args(ReplayBufferPPO),
    "TQC": ignore_extra_args(ReplayBufferTQC)
}

class Logger:
    def __init__(self, experiment_folder, config):
        if os.path.exists(experiment_folder):
            raise Exception('Experiment folder exists, apparent seed conflict!')
        os.makedirs(experiment_folder)
        self.metrics_file = experiment_folder / "metrics.json"
        self.energies_file = experiment_folder / "energies.json"
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

    def log_energies(self, metrics):
        with open(self.energies_file, 'a') as f:
            json.dump(metrics, f)
            f.write('\n')

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
    # Set env name
    args.env = args.db_path.split('/')[-1].split('.')[0]
    
    # Initialize logger
    logger = Logger(experiment_folder, args)
    
    # Initialize env
    env_kwargs = {
        'db_path': args.db_path,
        'timelimit': args.timelimit,
        'done_on_timelimit': args.done_on_timelimit,
        'sample_initial_conformations': True,
        'num_initial_conformations': args.num_initial_conformations,
        'inject_noise': args.inject_noise,
        'noise_std': args.noise_std,
        'remove_hydrogen': args.remove_hydrogen,
    }
    
    # Reward wrapper kwargs
    reward_wrapper_kwargs = {
        'minimize_on_every_step': args.minimize_on_every_step,
        'greedy': args.greedy,
        'remove_hydrogen': args.remove_hydrogen,
        'molecules_xyz_prefix': args.molecules_xyz_prefix,
        'M': args.M,
        'done_when_not_improved': args.done_when_not_improved
    }

    # Make parallel environments
    env = make_envs(env_kwargs, args.reward, reward_wrapper_kwargs, args.seed, args.num_processes)
    
    # Update kwargs and make an environment for evaluation
    env_kwargs['inject_noise'] = False
    eval_env = env_fn(**env_kwargs)
    reward_wrapper_kwargs.update({
        'env': eval_env,
        # 'done_when_not_improved': False
    })
    eval_env = reward_wrapper(args.reward, **reward_wrapper_kwargs)

    # Initialize action_scale scheduler and timelimit scheduler
    action_scale_scheduler = ActionScaleScheduler(action_scale_init=args.action_scale_init, 
                                                  action_scale_end=args.action_scale_end,
                                                  n_step_end=args.action_scale_n_step_end,
                                                  mode=args.action_scale_mode)
    if args.increment_timelimit:
        assert args.greedy, "Timelimit may be incremented during training only in greedy mode"
    timelimit_scheduler = TimelimitScheduler(timelimit_init=args.timelimit,
                                             step=args.timelimit_step,
                                             interval=args.timelimit_interval,
                                             constant=not args.increment_timelimit)
    
    use_ppo = args.algorithm == 'PPO'
    
    # Initialize replay buffer
    if  use_ppo:
        assert args.replay_buffer_size == args.update_frequency, \
            f"PPO algorithm requires replay_buffer_size == update_frequency, got {args.replay_buffer_size} and {args.update_frequency}"
    replay_buffer = replay_buffers[args.algorithm](
        device=DEVICE,
        n_processes=args.num_processes,
        max_size=args.replay_buffer_size
    )

    # Inititalize actor and critic
    schnet_args = {
        'n_interactions': args.n_interactions,
        'cutoff': args.cutoff,
        'n_gaussians': args.n_gaussians,
    }
    policy = policies[args.algorithm](
        schnet_args=schnet_args,
        out_embedding_size=args.out_embedding_size,
        action_scale_scheduler=action_scale_scheduler,
        n_nets=args.n_nets,
        n_quantiles=args.n_quantiles,
        tanh=args.tanh
    ).to(DEVICE)

    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets
    
    # Target entropy must differ for molecules of different sizes.
    # To achieve this we fix per-atom entropy instead of the entropy of the molecule.
    per_atom_target_entropy = 3 * (-1 + np.log([args.target_entropy_action_scale])).item()
    trainer = trainers[args.algorithm](
        policy=policy,
        # TQC arguments
        discount=args.discount,
        tau=args.tau,
        log_alpha=np.log([args.initial_alpha]).item(),
        actor_lr=args.actor_lr,
        critic_lr= args.critic_lr,
        alpha_lr=args.alpha_lr,
        top_quantiles_to_drop=top_quantiles_to_drop,
        per_atom_target_entropy=per_atom_target_entropy,
        batch_size=args.batch_size,
        actor_clip_value=args.actor_clip_value,
        critic_clip_value=args.critic_clip_value,
        # PPO arguments
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        lr=args.actor_lr,
        max_grad_norm=args.actor_clip_value,
        use_clipped_value_loss=args.use_clipped_value_loss
    )

    state = env.reset()
    episode_returns = np.zeros(args.num_processes)
    episode_mean_Q = np.zeros(args.num_processes)

    max_timesteps = int(args.max_timesteps) // args.num_processes

    policy.train()    
    full_checkpoints = [max_timesteps // 3, max_timesteps * 2 // 3, args.max_timesteps - 1]
    if args.load_model is not None:
        start_iter = int(args.load_model.split('/')[-1].split('_')[1]) + 1
        trainer.load(args.load_model)
        replay_buffer = pickle.load(open(f'{args.load_model}_replay', 'rb'))
    else:
        start_iter = 0

    for t in range(start_iter, max_timesteps):
        start = time.time()
        if use_ppo:
            update_condition = ((t + 1) % args.update_frequency) == 0
        else:
            update_condition = t >= args.batch_size // args.num_processes and t % args.update_frequency == 0
        action_scale_scheduler.update(t)
        
        # Update timelimit
        timelimit_scheduler.update(t)
        current_timelimit = timelimit_scheduler.get_timelimit()
        
        # Update timelimit in envs
        env.env_method("update_timelimit", current_timelimit)
        
        # Select next action
        with torch.no_grad():
            policy_out = policy.act(state)
            values, actions, log_probs = [x.cpu().numpy() for x in policy_out]
            if not use_ppo:
                # Mean over nets and quantiles to log
                values = values.mean(axis=(1, 2))
            else:
                # Remove extra dimension to store correctly
                values = values.squeeze(-1)
                log_probs = log_probs.squeeze(-1)

        next_state, rewards, dones, infos = env.step(actions)
        # Done on every step or at the end of the episode
        dones = [done or args.greedy for done in dones]
        episode_timesteps = env.env_method("get_env_step")
        ep_ends = [ep_t >= current_timelimit for ep_t in episode_timesteps]
        
        transition = [state, actions, next_state, rewards, dones]
        # if PPO then add log_prob, value and next value to RB
        if use_ppo:
            transition.extend([ep_ends, log_probs, values])
        replay_buffer.add(*transition)

        state = next_state
        episode_returns += rewards
        episode_mean_Q += values

        # Train agent after collecting sufficient data
        if update_condition:
            if use_ppo:
                with torch.no_grad():
                    next_value = policy.get_value(state).cpu()
                replay_buffer.compute_returns(next_value, args.discount, args.done_on_timelimit or args.greedy)
                step_metrics = trainer.update(replay_buffer)
            else:
                # Update TQC several times
                for _ in range(args.num_processes):
                    step_metrics = trainer.update(replay_buffer)
        else:
            step_metrics = dict()

        step_metrics['Timestamp'] = str(datetime.datetime.now())
        step_metrics['Action_scale'] = action_scale_scheduler.get_action_scale()
        step_metrics['Timelimit'] = current_timelimit
        step_metrics['Action_norm'] = calculate_action_norm(actions, state['_atom_mask']).item()
        
        # Update training statistics
        for i, (done, ep_end, info) in enumerate(zip(dones, ep_ends, infos)):
            logger.log_energies({
                'rdfkit_energy': info['rdkit_final_energy'],
                'dft_energy': info['dft_final_energy'],
                'dft_exception': info['dft_exception']
            })
            if done or (not args.greedy and ep_end):
                logger.update_evaluation_statistics(episode_timesteps[i],
                                                    episode_returns[i].item(),
                                                    episode_mean_Q[i].item(),
                                                    info['final_energy'],
                                                    info['final_rl_energy'],
                                                    info['not_converged'])
                episode_returns[i] = 0
                episode_mean_Q[i] = 0

        # If timelimit is reached or a real done comes from the environement reset the environment 
        envs_to_reset = [i for i, (done, ep_end) in enumerate(zip(dones, ep_ends)) if ep_end or (done and not args.greedy)]
        # Get new states and remove extra dimension
        reset_states = [{k:v.squeeze() for k, v in s.items()} for s in env.env_method("reset", indices=envs_to_reset)]

        # Recollate state_batch after resets as atomic numbers might have changed.
        # Execute only if at least one env has reset.
        if len(envs_to_reset) > 0:
            state = recollate_batch(state, envs_to_reset, reset_states)

        # Print update time
        print(time.time() - start)

        # Evaluate episode
        if (t + 1) % (args.eval_freq // args.num_processes) == 0:
            step_metrics['Total_timesteps'] = (t + 1) * args.num_processes
            step_metrics.update(
                eval_policy(policy, eval_env, current_timelimit, args.n_eval_runs,
                            args.n_explore_runs, args.reward == "rdkit",
                            args.evaluate_multiple_timesteps)
            )
            logger.log(step_metrics)

        if t in full_checkpoints and args.save_checkpoints:
            trainer_save_name = f'{experiment_folder}/iter_{t}'
            trainer.save(trainer_save_name)
            #with open(f'{experiment_folder}/iter_{t}_replay', 'wb') as outF:
            #    pickle.dump(replay_buffer, outF)
            # Remove previous checkpoint?
        elif (t + 1) % (args.light_checkpoint_freq // args.num_processes) == 0 and args.save_checkpoints:
            trainer_save_name = f'{experiment_folder}/iter_{t}'
            trainer.light_save(trainer_save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Algoruthm
    parser.add_argument("--algorithm", default='TQC', choices=['TQC', 'PPO'])
    # Env args
    parser.add_argument("--num_processes", default=1, type=int, help="Number of copies of env to run in parallel")
    parser.add_argument("--db_path", default="env/data/malonaldehyde.db", type=str, help="Path to molecules database")
    parser.add_argument("--num_initial_conformations", default=50000, type=int, help="Number of initial molecule conformations to sample from DB")
    parser.add_argument("--sample_initial_conformation", default=False, type=bool, help="Sample new conformation for every seed")
    parser.add_argument("--inject_noise", type=bool, default=False, help="Whether to inject random noise into initial states")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Std of the injected noise")
    parser.add_argument("--remove_hydrogen", type=bool, default=False, help="Whether to remove hydrogen atoms from the molecule")
    # Timelimit args
    parser.add_argument("--timelimit", default=100, type=int, help="Timelimit for MD env")
    parser.add_argument("--increment_timelimit", default=False, type=bool, help="Whether to increment timelimit during training")
    parser.add_argument("--timelimit_step", default=10, type=int, help="By which number to increment timelimit")
    parser.add_argument("--timelimit_interval", default=150000, type=int, help="How often to increment timelimit")
    parser.add_argument("--greedy", default=False, type=bool, help="Returns done on every step independent of the timelimit")
    parser.add_argument("--done_on_timelimit", type=bool, default=False, help="Env returns done when timelimit is reached")
    parser.add_argument("--done_when_not_improved", type=bool, default=False, help="Return done if energy has not improved")
    # Reward args
    parser.add_argument("--reward", choices=["rdkit", "dft"], default="rdkit", help="How the energy is calculated")
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
    # Policy args
    parser.add_argument("--out_embedding_size", default=128, type=int, help="Output embedding size for policy")
    parser.add_argument("--tanh", choices=["before_projection", "after_projection"], help="Whether to put tanh() before projection operator or after")
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    # Eval args
    parser.add_argument("--eval_freq", default=1e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--n_eval_runs", default=10, type=int, help="Number of evaluation episodes")
    parser.add_argument("--n_explore_runs", default=5, type=int, help="Number of exploration episodes during evaluation")
    parser.add_argument("--evaluate_multiple_timesteps", default=False, type=bool, help="Evaluate at multiple timesteps")
    # TQC args
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--actor_lr", default=3e-4, type=float, help="Actor learning rate")
    parser.add_argument("--actor_clip_value", default=None, help="Clipping value for actor gradients")
    parser.add_argument("--critic_lr", default=3e-4, type=float, help="Critic learning rate")
    parser.add_argument("--critic_clip_value", default=None, help="Clipping value for critic gradients")
    parser.add_argument("--alpha_lr", default=3e-4, type=float, help="Alpha learning rate")
    parser.add_argument("--initial_alpha", default=1.0, type=float, help="Initial value for alpha")
    # PPO args
    parser.add_argument("--clip_param", default=0.2, type=float)
    parser.add_argument("--ppo_epoch", default=4, type=int)
    parser.add_argument("--num_mini_batch", default=16, type=int)
    parser.add_argument("--value_loss_coef", default=0.5, type=float)
    parser.add_argument("--entropy_coef", default=0.01, type=float)
    parser.add_argument("--use_clipped_value_loss", default=True, type=bool)
    # Other args
    parser.add_argument("--exp_name", required=True, type=str, help="Name of the experiment")
    parser.add_argument("--replay_buffer_size", default=int(1e5), type=int, help="Size of replay buffer")
    parser.add_argument("--update_frequency", default=1, type=int, help="How often agent is updated")
    parser.add_argument("--max_timesteps", default=1e6, type=int, help="Max time steps to run environment")
    parser.add_argument("--seed", default=None, type=int, help="Random seed")
    parser.add_argument("--light_checkpoint_freq", type=int, default=100000, help="How often light checkpoint is saved")
    parser.add_argument("--save_checkpoints", type=bool, default=False, help="Save light and full checkpoints")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load the model from")
    parser.add_argument("--log_dir", default='.', help="Directory where runs are saved")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
     # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    #args.git_sha = get_current_gitsha()

    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    exp_folder = log_dir / f'{args.exp_name}_{start_time}_{args.seed}'

    main(args, exp_folder)
