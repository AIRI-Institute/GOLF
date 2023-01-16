import datetime
import glob
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from schnetpack.nn import get_cutoff_by_string

from env.make_envs import make_envs
from rl import DEVICE
from rl.actor_critics.gd import GDPolicy
from rl.actor_critics.ppo import PPOPolicy
from rl.actor_critics.tqc import TQCPolicy
from rl.algos.gd import GD
from rl.algos.ppo import PPO
from rl.algos.tqc import TQC
from rl.eval import eval_policy_dft, eval_policy_rdkit
from rl.replay_buffer import ReplayBufferPPO, ReplayBufferTQC, ReplayBufferGD
from rl.utils import (TimelimitScheduler,
                      calculate_action_norm, recollate_batch, 
                      calculate_molecule_metrics)
from utils.arguments import get_args
from utils.logging import Logger
from utils.utils import ignore_extra_args

policies = {
    "PPO": ignore_extra_args(PPOPolicy),
    "TQC": ignore_extra_args(TQCPolicy),
    "SAC": ignore_extra_args(TQCPolicy),
    "GD": ignore_extra_args(GDPolicy),
}

trainers = {
    "PPO": ignore_extra_args(PPO),
    "TQC": ignore_extra_args(TQC),
    "SAC": ignore_extra_args(TQC),
    "GD": ignore_extra_args(GD),
}

replay_buffers = {
    "PPO": ignore_extra_args(ReplayBufferPPO),
    "TQC": ignore_extra_args(ReplayBufferTQC),
    "SAC": ignore_extra_args(ReplayBufferTQC),
    "GD": ignore_extra_args(ReplayBufferGD),
}

eval_function = {
    "rdkit": ignore_extra_args(eval_policy_rdkit),
    "dft": ignore_extra_args(eval_policy_dft)
}

def main(args, experiment_folder):
    # Set env name
    args.env = args.db_path.split('/')[-1].split('.')[0]
    
    # Initialize logger
    logger = Logger(experiment_folder, args)
    
    use_ppo = args.algorithm == 'PPO'
    use_gd = args.algorithm == 'GD'

    # Initialize envs
    env, eval_env = make_envs(args)

    if args.increment_timelimit:
        assert args.greedy, "Timelimit may be incremented during training only in greedy mode"
    timelimit_scheduler = TimelimitScheduler(timelimit_init=args.timelimit,
                                             step=args.timelimit_step,
                                             interval=args.timelimit_interval,
                                             constant=not args.increment_timelimit)
    
    # Initialize replay buffer
    if use_ppo:
        assert args.replay_buffer_size == args.update_frequency, \
            f"PPO algorithm requires replay_buffer_size == update_frequency, got {args.replay_buffer_size} and {args.update_frequency}"
        max_size = args.replay_buffer_size // args.n_parallel
    else:
        max_size = args.replay_buffer_size
    replay_buffer = replay_buffers[args.algorithm](
        device=DEVICE,
        n_processes=args.n_parallel,
        max_size=max_size
    )

    # Inititalize actor and critic
    backbone_args = {
        'n_interactions': args.n_interactions,
        'cutoff': args.cutoff,
        'n_gaussians': args.n_rbf,
        'n_rbf':  args.n_rbf,
        'use_cosine_between_vectors': args.use_cosine_between_vectors
    }
    policy = policies[args.algorithm](
        backbone=args.backbone,
        backbone_args=backbone_args,
        generate_action_type=args.generate_action_type,
        critic_type=args.algorithm,
        out_embedding_size=args.out_embedding_size,
        cutoff_type=args.cutoff_type,
        summation_order=args.summation_order,
        use_activation=args.use_activation,
        n_nets=args.n_nets,
        m_nets=args.m_nets,
        n_quantiles=args.n_quantiles,
        limit_actions=args.limit_actions,
        action_scale=args.action_scale,
        action_norm_limit=args.action_norm_limit,
    ).to(DEVICE)

    # Initialize cutoff network for logging purposes
    cutoff_network = get_cutoff_by_string(args.cutoff_type)(args.cutoff).to(DEVICE)

    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.m_nets
    
    # Target entropy must differ for molecules of different sizes.
    # To achieve this we fix per-atom entropy instead of the entropy of the molecule.
    per_atom_target_entropy = 3 * (-1 + np.log([args.target_entropy_action_scale])).item()
    trainer = trainers[args.algorithm](
        policy=policy,
        # TQC arguments
        critic_type=args.algorithm,
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
        lr_scheduler=args.lr_scheduler,
        total_steps=args.max_timesteps,
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
    if use_gd:
        current_energy = np.array(env.initial_energy[args.reward], dtype=np.float32)
        forces = [np.array(force, dtype=np.float32) for force in env.force[args.reward]]
        replay_buffer.add(state, None, forces, current_energy, None)

    episode_returns = np.zeros(args.n_parallel)
    episode_Q = np.zeros(args.n_parallel)


    if args.load_model is not None:
        start_iter = int(args.load_model.split('/')[-1].split('_')[-1]) // args.n_parallel + 1 
        trainer.load(args.load_model)
        replay_buffer = pickle.load(open(f'{args.load_model}_replay', 'rb'))
    else:
        start_iter = 0

    policy.train()
    max_timesteps = int(args.max_timesteps) // args.n_parallel


    for t in range(start_iter, max_timesteps):
        start = time.perf_counter()
        if use_ppo:
            update_condition = (((t + 1) * args.n_parallel) % args.update_frequency) == 0
        else:
            update_condition = (t + 1) >= args.batch_size // args.n_parallel and (t + 1) % args.update_frequency == 0
            update_actor_condition = (t + 1) > args.pretrain_critic // args.n_parallel
        
        # Update timelimit
        timelimit_scheduler.update(t)
        current_timelimit = timelimit_scheduler.get_timelimit()
        
        # Update timelimit in envs
        env.unwrapped.update_timelimit(current_timelimit)
        
        # Select next action
        if use_gd:
            actions = policy.act(state)['action'].cpu().numpy()
            values = np.zeros(actions.shape[0], dtype=np.float32)
        else:
            with torch.no_grad():
                policy_out = policy.act(state)
                values, actions, log_probs = [x.cpu().numpy() for x in policy_out]
                if args.algorithm == "TQC":
                    # Mean over nets and quantiles
                    values = values.mean(axis=(1, 2))
                elif args.algorithm == "SAC":
                    values = values.mean(axis=1)
                elif use_ppo:
                    # Remove extra dimension to store correctly
                    values = values.squeeze(-1)
                    log_probs = log_probs.squeeze(-1)

        next_state, rewards, dones, info = env.step(actions)
        # Done on every step or at the end of the episode
        dones = [done or args.greedy for done in dones]
        episode_timesteps = env.unwrapped.get_env_step()
        ep_ends = [ep_t >= current_timelimit for ep_t in episode_timesteps]
        
        transition = [state, actions, next_state, rewards, dones]
        # if PPO then add log_prob, value and next value to RB
        if use_ppo:
            transition.extend([ep_ends, log_probs, values])
        elif use_gd:
            current_energy = np.array(env.initial_energy[args.reward], dtype=np.float32)
            forces = [np.array(force, dtype=np.float32) for force in env.force[args.reward]]
            transition[2] = forces
            transition[3] = current_energy
        replay_buffer.add(*transition)

        # Estimate average number of atoms inside cutoff radius
        # and min/avg/max distance between atoms for both
        # state and next_state before moving to the next state
        molecule_metrics = calculate_molecule_metrics(state, next_state, cutoff_network)

        state = next_state
        episode_returns += rewards
        # Set episode Q if t == 1
        episode_Q = [value if t == 1 else Q for value, t, Q in zip(values, episode_timesteps, episode_Q)]

        # Train agent after collecting sufficient data
        if update_condition:
            if use_ppo:
                with torch.no_grad():
                    next_value = policy.get_value(state).cpu()
                replay_buffer.compute_returns(next_value, args.discount, args.done_on_timelimit or args.greedy)
                step_metrics = trainer.update(replay_buffer)
            else:
                # Update TQC several times
                for _ in range(args.n_parallel):
                    step_metrics = trainer.update(replay_buffer, update_actor_condition, args.greedy)
        else:
            step_metrics = dict()

        step_metrics['Timestamp'] = str(datetime.datetime.now())
        step_metrics['Timelimit'] = current_timelimit
        step_metrics['Action_norm'] = calculate_action_norm(actions, state['_atom_mask']).item()
        
        # Calculate average number of pairs of atoms too close together
        # in env before and after processing
        step_metrics['Molecule/num_bad_pairs_before'] = info['total_bad_pairs_before_process']
        step_metrics['Molecule/num_bad_pairs_after'] = info['total_bad_pairs_after_process']
        step_metrics.update(molecule_metrics)
        
        # Update training statistics
        for i, (done, ep_end) in enumerate(zip(dones, ep_ends)):
            if done or (not args.greedy and ep_end):
                logger.update_evaluation_statistics(episode_timesteps[i],
                                                    episode_returns[i].item(),
                                                    episode_Q[i].item(),
                                                    info['final_energy'][i],
                                                    info['final_rl_energy'][i],
                                                    info['threshold_exceeded_pct'][i],
                                                    info['not_converged'][i])
                episode_returns[i] = 0

        # If timelimit is reached or a real done comes from the environement reset the environment 
        envs_to_reset = [i for i, (done, ep_end) in enumerate(zip(dones, ep_ends)) if ep_end or (done and not args.greedy)]

        # Recollate state_batch after resets as atomic numbers might have changed.
        # Execute only if at least one env has reset.
        if len(envs_to_reset) > 0:
            reset_states = env.reset(indices=envs_to_reset)
            state = recollate_batch(state, envs_to_reset, reset_states)
            if use_gd:
                energies = np.array([env.initial_energy[args.reward][idx] for idx in envs_to_reset], dtype=np.float32)
                forces = [np.array(env.force[args.reward][idx], dtype=np.float32) for idx in envs_to_reset]
                replay_buffer.add(reset_states, None, forces, energies, None)

        # Print update time
        # print(time.perf_counter() - start)

        # Evaluate episode
        if (t + 1) % (args.eval_freq // args.n_parallel) == 0:
            step_metrics['Total_timesteps'] = (t + 1) * args.n_parallel
            step_metrics['FPS'] = args.n_parallel / (time.perf_counter() - start)
            step_metrics.update(
                eval_function[args.reward](
                    actor=policy,
                    env=eval_env,
                    max_timestamps=current_timelimit,
                    eval_episodes=args.n_eval_runs,
                    n_explore_runs=args.n_explore_runs,
                    evaluate_multiple_timesteps=args.evaluate_multiple_timesteps
                )
            )
            logger.log(step_metrics)

        # Save checkpoints
        if (t + 1) % (args.full_checkpoint_freq // args.n_parallel) == 0 and args.save_checkpoints:
            # Remove previous checkpoint
            old_checkpoint_files = glob.glob(f'{experiment_folder}/full_cp_iter*')
            for cp_file in old_checkpoint_files:
                os.remove(cp_file)
            
            # Save new checkpoint
            trainer_save_name = f'{experiment_folder}/full_cp_iter_{t + 1}'
            trainer.save(trainer_save_name)
            if not use_ppo:
                with open(f'{experiment_folder}/full_cp_iter_{t + 1}_replay', 'wb') as outF:
                    pickle.dump(replay_buffer, outF)

        if (t + 1) % (args.light_checkpoint_freq // args.n_parallel) == 0 and args.save_checkpoints and not use_ppo:
            trainer_save_name = f'{experiment_folder}/light_cp_iter_{t + 1}'
            trainer.light_save(trainer_save_name)


if __name__ == "__main__":
    args = get_args()

    log_dir = Path(args.log_dir)
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
     # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    #args.git_sha = get_current_gitsha()

    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    if args.load_model is not None:
        assert os.path.exists(f'{args.load_model}_actor'), "Checkpoint not found!"
        exp_folder = log_dir / args.load_model.split('/')[-2]
    else:
        exp_folder = log_dir / f'{args.exp_name}_{start_time}_{args.seed}'

    main(args, exp_folder)
