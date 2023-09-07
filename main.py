import copy
import datetime
import glob
import math
import os
import pickle
import random
import time
from pathlib import Path

from ase.db import connect
import numpy as np
import torch

from AL import DEVICE
from AL.AL_trainer import AL
from AL.eval import eval_policy_dft, eval_policy_rdkit
from AL.make_policies import make_policies
from AL.make_saver import make_saver
from AL.replay_buffer import ReplayBuffer, fill_initial_replay_buffer
from AL.utils import calculate_action_norm, recollate_batch
from env.make_envs import make_envs
from utils.arguments import get_args
from utils.logging import Logger
from utils.utils import ignore_extra_args

eval_function = {
    "rdkit": ignore_extra_args(eval_policy_rdkit),
    "dft": ignore_extra_args(eval_policy_dft),
}

REWARD_THRESHOLD = -100


def main(args, experiment_folder):
    # Set env name
    args.env = args.db_path.split("/")[-1].split(".")[0]

    # Initialize logger
    logger = Logger(experiment_folder, args)

    # Initialize envs
    env, eval_env = make_envs(args)

    # Initialize replay buffer.
    atomrefs = None
    initial_replay_buffer = None
    # First, initialize a RB with initial conformations
    if args.reward == "dft":
        # Read atomization energy from the database
        with connect(args.db_path) as conn:
            if "atomrefs" in conn.metadata and args.subtract_atomization_energy:
                atomrefs = conn.metadata["atomrefs"]["energy"]
        assert (
            args.subtract_atomization_energy and atomrefs
        ), "Attempting to train with no atomization energy subtraction\
            will likely result in the divergence of the model"

    # Initialize a fixed replay buffer with conformations from the database
    print("Filling replay buffer with initial conformations...")
    initial_replay_buffer = fill_initial_replay_buffer(DEVICE, args, atomrefs)
    print(f"Done! RB size: {initial_replay_buffer.size}")

    replay_buffer = ReplayBuffer(
        device=DEVICE,
        max_size=args.max_oracle_steps,
        atomrefs=atomrefs,
        initial_RB=initial_replay_buffer,
        initial_conf_pct=args.initial_conf_pct,
    )

    # Inititalize policy and eval policy
    policy, eval_policy = make_policies(env, eval_env, args)

    # Initialize experience saver
    experience_saver = make_saver(
        args,
        env=env,
        replay_buffer=replay_buffer,
        actor=policy.actor,
        reward_thresh=REWARD_THRESHOLD,
    )

    # Initialize trainer
    trainer = AL(
        policy=policy,
        lr=args.lr,
        batch_size=args.batch_size,
        clip_value=args.clip_value,
        lr_scheduler=args.lr_scheduler,
        energy_loss_coef=args.energy_loss_coef,
        force_loss_coef=args.force_loss_coef,
        total_steps=args.max_oracle_steps * args.utd_ratio,
        optimizer_name=args.optimizer,
    )

    state = env.reset()
    # Set initial states in Policy
    policy.reset(state)

    episode_returns = np.zeros(args.n_parallel)

    if args.load_model is not None:
        start_iter = (
            int(args.load_model.split("/")[-1].split("_")[-1]) // args.n_parallel + 1
        )
        trainer.load(args.load_model)
        replay_buffer = pickle.load(open(f"{args.load_model}_replay", "rb"))
    else:
        start_iter = 0
        if args.load_baseline is not None:
            trainer.light_load(args.load_baseline)

    policy.train()

    # Set training flag to False (for dft reward only)
    train_model_flag = False

    # Set evaluation/save flags to False (for dft reward only).
    # Flag is set to True every time new experience is added
    # to replay buffer. This is done to avoid multiple
    # evaluations of the same model
    eval_model_flag = False
    full_save_flag = False
    light_save_flag = False

    # Train until the number of conformations in
    # replay buffer is less than max_oracle_steps
    while not replay_buffer.replay_buffer_full:
        start = time.perf_counter()
        update_condition = replay_buffer.size >= args.batch_size

        # Get current timesteps
        episode_timesteps = env.unwrapped.get_env_step()

        if not args.store_only_initial_conformations:
            # Select next action
            actions = policy.act(episode_timesteps)["action"].cpu().numpy()
            print("policy.act() time: {:.4f}".format(time.perf_counter() - start))

            # If action contains non finites then reset everything and continue
            if not np.isfinite(actions).all():
                state = env.reset()
                policy.reset(state)
                episode_returns = np.zeros(args.n_parallel)
                continue

            next_state, rewards, dones, info = env.step(actions)

            if args.reward == "dft":
                # If task queue is full wait for all tasks to finish and store data to RB
                if env.dft_oracle.task_queue_full_flag:
                    (
                        states,
                        energies,
                        forces,
                        episode_total_delta_energies,
                    ) = env.dft_oracle.get_data()
                    replay_buffer.add(states, forces, energies)

                    assert len(episode_returns) == len(episode_total_delta_energies)
                    logger.update_dft_return_statistics(episode_total_delta_energies)

                    # After new data has been added to replay buffer reset all flags
                    train_model_flag = True
                    eval_model_flag = True
                    full_save_flag = True
                    light_save_flag = True
            else:
                experience_saver(next_state, rewards, dones)
                episode_returns += rewards

            # Move to next state
            state = next_state
        else:
            dones = np.stack([True for _ in range(args.n_parallel)])

        if update_condition and (args.reward != "dft" or train_model_flag):
            # Train agent after collecting sufficient data
            prev_start = time.perf_counter()
            for update_num in range(args.n_parallel * args.utd_ratio):
                step_metrics = trainer.update(replay_buffer)
                new_start = time.perf_counter()
                print(
                    "policy.train {} time: {:.4f}".format(
                        update_num, new_start - prev_start
                    )
                )
                prev_start = new_start
            train_model_flag = False
        else:
            step_metrics = dict()

        step_metrics["Timestamp"] = str(datetime.datetime.now())
        step_metrics["RB_size"] = replay_buffer.size

        if not args.store_only_initial_conformations:
            step_metrics["Action_norm"] = calculate_action_norm(
                actions, env.get_atoms_num_cumsum()
            ).item()
            # Calculate average number of pairs of atoms too close together
            # in env before and after processing
            step_metrics["Molecule/num_bad_pairs_before"] = info[
                "total_bad_pairs_before_processing"
            ]
            step_metrics["Molecule/num_bad_pairs_after"] = info[
                "total_bad_pairs_after_processing"
            ]
            # Update training statistics
            for i, done in enumerate(dones):
                if done:
                    logger.update_evaluation_statistics(
                        episode_timesteps[i] + 1,
                        episode_returns[i].item(),
                        info["final_energy"][i],
                    )
                    episode_returns[i] = 0

        # If the episode is terminated
        envs_to_reset = [i for i, done in enumerate(dones) if done]

        # Recollate state_batch after resets.
        # Execute only if at least one env has reset.
        if len(envs_to_reset) > 0:
            reset_states = env.reset(indices=envs_to_reset)
            state = recollate_batch(state, envs_to_reset, reset_states)
            # Reset initial states in policy
            policy.reset(reset_states, indices=envs_to_reset)

        # Print iteration time
        print("Full iteration time: {:.4f}".format(time.perf_counter() - start))

        # Evaluate episode
        if (args.reward != "dft" or eval_model_flag) and (
            replay_buffer.size // args.n_parallel
        ) % math.ceil(args.eval_freq / float(args.n_parallel)) == 0:
            print(f"Evaluation at step {replay_buffer.size}...")
            # Update eval policy
            eval_policy.actor = copy.deepcopy(policy.actor)
            step_metrics["Total_timesteps"] = replay_buffer.size
            step_metrics["Total_training_steps"] = replay_buffer.size * args.utd_ratio
            step_metrics["FPS"] = args.n_parallel / (time.perf_counter() - start)
            if not args.store_only_initial_conformations or args.reward == "rdkit":
                step_metrics.update(
                    eval_function[args.reward](
                        actor=eval_policy,
                        env=eval_env,
                        eval_episodes=args.n_eval_runs,
                        eval_termination_mode=args.eval_termination_mode,
                    )
                )
            logger.log(step_metrics)

            # Prevent evaluations until new data is added to replay buffer
            eval_model_flag = False

        # Save checkpoints
        if (
            (args.reward != "dft" or full_save_flag)
            and (replay_buffer.size // args.n_parallel)
            % (args.full_checkpoint_freq // args.n_parallel)
            == 0
            and args.save_checkpoints
        ):
            # Remove previous checkpoint
            old_checkpoint_files = glob.glob(f"{experiment_folder}/full_cp_iter*")
            for cp_file in old_checkpoint_files:
                os.remove(cp_file)

            # Save new checkpoint
            save_t = replay_buffer.size
            trainer_save_name = f"{experiment_folder}/full_cp_iter_{save_t}"
            trainer.save(trainer_save_name)
            with open(
                f"{experiment_folder}/full_cp_iter_{save_t}_replay", "wb"
            ) as outF:
                pickle.dump(replay_buffer, outF)

            # Prevent checkpoint saving until new data is added to replay buffer
            full_save_flag = False

        if (
            (args.reward != "dft" or light_save_flag)
            and (replay_buffer.size // args.n_parallel)
            % (args.light_checkpoint_freq // args.n_parallel)
            == 0
            and args.save_checkpoints
        ):
            save_t = replay_buffer.size
            trainer_save_name = f"{experiment_folder}/light_cp_iter_{save_t}"
            trainer.light_save(trainer_save_name)

            # Prevent checkpoint saving until new data is added to replay buffer
            light_save_flag = False


if __name__ == "__main__":
    args = get_args()

    log_dir = Path(args.log_dir)
    if args.seed is None:
        args.seed = random.randint(0, 1000000)

    # Check hyperparameters
    if args.store_only_initial_conformations:
        assert args.timelimit_train == 1

    if args.reward == "rdkit":
        assert not args.subtract_atomization_energy

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # args.git_sha = get_current_gitsha()

    start_time = datetime.datetime.strftime(
        datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S"
    )
    if args.load_model is not None:
        assert os.path.exists(f"{args.load_model}_actor"), "Checkpoint not found!"
        exp_folder = log_dir / args.load_model.split("/")[-2]
    else:
        exp_folder = log_dir / f"{args.exp_name}_{start_time}_{args.seed}"

    main(args, exp_folder)
