import datetime
import glob
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import schnetpack

from env.make_envs import make_envs
from AL import DEVICE
from AL.AL_actor import ALPolicy
from AL.AL_trainer import AL
from AL.eval import eval_policy_dft, eval_policy_rdkit
from AL.replay_buffer import ReplayBufferGD
from AL.utils import calculate_action_norm, recollate_batch, get_cutoff_by_string
from utils.arguments import get_args
from utils.logging import Logger
from utils.utils import ignore_extra_args

eval_function = {
    "rdkit": ignore_extra_args(eval_policy_rdkit),
    "dft": ignore_extra_args(eval_policy_dft)
}


def main(args, experiment_folder):
    # Set env name
    args.env = args.db_path.split('/')[-1].split('.')[0]

    # Initialize logger
    logger = Logger(experiment_folder, args)

    # Initialize envs
    env, eval_env = make_envs(args)

    # Initialize replay buffer
    replay_buffer = ReplayBufferGD(
        device=DEVICE,
        max_size=args.replay_buffer_size
    )

    if args.store_only_initial_conformations:
        assert args.timelimit_train == 1

    # Inititalize actor and critic
    backbone_args = {
        'n_interactions': args.n_interactions,
        'n_atom_basis': args.n_atom_basis,
        'radial_basis': schnetpack.nn.BesselRBF(n_rbf=args.n_rbf, cutoff=args.cutoff),
        'cutoff_fn': get_cutoff_by_string('cosine')(args.cutoff),
    }
    policy = ALPolicy(
        backbone=args.backbone,
        backbone_args=backbone_args,
        action_scale_sheduler=args.action_scale_sheduler,
        action_scale=args.action_scale,
        action_norm_limit=args.action_norm_limit,
    ).to(DEVICE)

    trainer = AL(
        policy=policy,
        lr=args.lr,
        batch_size=args.batch_size,
        clip_value=args.clip_value,
        lr_scheduler=args.lr_scheduler,
        energy_loss_coef=args.energy_loss_coef,
        force_loss_coef=args.force_loss_coef,
        total_steps=args.max_timesteps,
    )

    state = env.reset()

    # Save initial state in replay buffer
    current_energy = np.array(env.initial_energy[args.reward], dtype=np.float32)
    forces = [np.array(force, dtype=np.float32) for force in env.force[args.reward]]
    replay_buffer.add(state, forces, current_energy)

    episode_returns = np.zeros(args.n_parallel)

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
        update_condition = (t + 1) >= args.batch_size // args.n_parallel and (t + 1)

        # Get current timesteps
        episode_timesteps = env.unwrapped.get_env_step()
        
        # Select next action
        actions = policy.act(state, episode_timesteps)['action'].cpu().numpy()
        next_state, rewards, dones, info = env.step(actions)

        if not args.store_only_initial_conformations:
            current_energy = np.array(env.initial_energy[args.reward], dtype=np.float32)
            forces = [np.array(force, dtype=np.float32) for force in env.force[args.reward]]
            transition = [next_state, forces, current_energy]
            replay_buffer.add(*transition)

        state = next_state
        episode_returns += rewards

        # Train agent after collecting sufficient data
        if update_condition:
            for _ in range(args.n_parallel):
                step_metrics = trainer.update(replay_buffer)
        else:
            step_metrics = dict()

        step_metrics['Timestamp'] = str(datetime.datetime.now())
        step_metrics['Action_norm'] = calculate_action_norm(actions, env.get_atoms_num_cumsum()).item()

        # Calculate average number of pairs of atoms too close together
        # in env before and after processing
        step_metrics['Molecule/num_bad_pairs_before'] = info['total_bad_pairs_before_process']
        step_metrics['Molecule/num_bad_pairs_after'] = info['total_bad_pairs_after_process']

        # Update training statistics
        for i, done in enumerate(dones):
            if done:
                logger.update_evaluation_statistics(episode_timesteps[i] + 1,
                                                    episode_returns[i].item(),
                                                    info['final_energy'][i],
                                                    info['final_rl_energy'][i],
                                                    info['threshold_exceeded_pct'][i],
                                                    info['not_converged'][i])
                episode_returns[i] = 0

        # If the episode is terminated
        envs_to_reset = [i for i, done in enumerate(dones) if done]

        # Recollate state_batch after resets as atomic numbers might have changed.
        # Execute only if at least one env has reset.
        if len(envs_to_reset) > 0:
            reset_states = env.reset(indices=envs_to_reset)
            state = recollate_batch(state, envs_to_reset, reset_states)
            energies = np.array([env.initial_energy[args.reward][idx] for idx in envs_to_reset], dtype=np.float32)
            forces = [np.array(env.force[args.reward][idx], dtype=np.float32) for idx in envs_to_reset]
            replay_buffer.add(reset_states, forces, energies)

        print(time.perf_counter() - start)

        # Evaluate episode
        if (t + 1) % (args.eval_freq // args.n_parallel) == 0:
            step_metrics['Total_timesteps'] = (t + 1) * args.n_parallel
            step_metrics['FPS'] = args.n_parallel / (time.perf_counter() - start)
            step_metrics.update(
                eval_function[args.reward](
                    actor=policy,
                    env=eval_env,
                    eval_episodes=args.n_eval_runs,
                    evaluate_multiple_timesteps=args.evaluate_multiple_timesteps,
                    eval_termination_mode=args.eval_termination_mode,
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
            save_t = (t + 1) * args.n_parallel
            trainer_save_name = f'{experiment_folder}/full_cp_iter_{save_t}'
            trainer.save(trainer_save_name)
            with open(f'{experiment_folder}/full_cp_iter_{save_t}_replay', 'wb') as outF:
                pickle.dump(replay_buffer, outF)

        if (t + 1) % (args.light_checkpoint_freq // args.n_parallel) == 0 and args.save_checkpoints:
            save_t = (t + 1) * args.n_parallel
            trainer_save_name = f'{experiment_folder}/light_cp_iter_{save_t}'
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
    # args.git_sha = get_current_gitsha()

    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    if args.load_model is not None:
        assert os.path.exists(f'{args.load_model}_actor'), "Checkpoint not found!"
        exp_folder = log_dir / args.load_model.split('/')[-2]
    else:
        exp_folder = log_dir / f'{args.exp_name}_{start_time}_{args.seed}'

    main(args, exp_folder)
