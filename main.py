import datetime
import numpy as np
import pickle
import random
import time
import torch

from pathlib import Path

from env.moldynamics_env import env_fn
from env.wrappers import RewardWrapper
from env.make_envs import make_envs

from rl import DEVICE
from rl.tqc import TQC
from rl.ppo import PPO
from rl.actor_critic_tqc import TQCPolicy
from rl.actor_critic_ppo import PPOPolicy
from rl.replay_buffer import ReplayBufferPPO, ReplayBufferTQC
from rl.utils import recollate_batch, calculate_action_norm
from rl.utils import ActionScaleScheduler, TimelimitScheduler
from rl.eval import eval_policy

from utils.logging import Logger
from utils.arguments import get_args
from utils.utils import ignore_extra_args

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
        'dft': args.reward == "dft",
        'minimize_on_every_step': args.minimize_on_every_step,
        'greedy': args.greedy,
        'remove_hydrogen': args.remove_hydrogen,
        'molecules_xyz_prefix': args.molecules_xyz_prefix,
        'M': args.M,
        'done_when_not_improved': args.done_when_not_improved
    }

    # Make parallel environments
    env = make_envs(env_kwargs, reward_wrapper_kwargs, args.seed, args.num_processes)
    
    # Update kwargs and make an environment for evaluation
    env_kwargs['inject_noise'] = False
    eval_env = env_fn(**env_kwargs)
    reward_wrapper_kwargs.update({
        'env': eval_env,
    })
    eval_env = RewardWrapper(**reward_wrapper_kwargs)

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
    if use_ppo:
        assert args.replay_buffer_size == args.update_frequency, \
            f"PPO algorithm requires replay_buffer_size == update_frequency, got {args.replay_buffer_size} and {args.update_frequency}"
    replay_buffer = replay_buffers[args.algorithm](
        device=DEVICE,
        n_processes=args.num_processes,
        max_size=args.replay_buffer_size
    )

    # Inititalize actor and critic
    backbone_args = {
        'n_interactions': args.n_interactions,
        'cutoff': args.cutoff,
        'n_gaussians': args.n_gaussians,
        'n_rbf':  args.n_gaussians
    }
    policy = policies[args.algorithm](
        backbone=args.backbone,
        backbone_args=backbone_args,
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
            update_condition = (t + 1) >= args.batch_size // args.num_processes and (t + 1) % args.update_frequency == 0
            update_actor_condition = (t + 1) > args.pretrain_critic // args.num_processes
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
                    step_metrics = trainer.update(replay_buffer, update_actor_condition)
        else:
            step_metrics = dict()

        step_metrics['Timestamp'] = str(datetime.datetime.now())
        step_metrics['Action_scale'] = action_scale_scheduler.get_action_scale()
        step_metrics['Timelimit'] = current_timelimit
        step_metrics['Action_norm'] = calculate_action_norm(actions, state['_atom_mask']).item()
        
        # Update training statistics
        for i, (done, ep_end, info) in enumerate(zip(dones, ep_ends, infos)):
            if done or (not args.greedy and ep_end):
                logger.update_evaluation_statistics(episode_timesteps[i],
                                                    episode_returns[i].item(),
                                                    episode_mean_Q[i].item(),
                                                    info['final_energy'],
                                                    info['final_rl_energy'],
                                                    info['threshold_exceeded_pct'],
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

        if (t + 1) in full_checkpoints and args.save_checkpoints:
            trainer_save_name = f'{experiment_folder}/iter_{t}'
            trainer.save(trainer_save_name)
            #with open(f'{experiment_folder}/iter_{t}_replay', 'wb') as outF:
            #    pickle.dump(replay_buffer, outF)
            # Remove previous checkpoint?
        elif (t + 1) % (args.light_checkpoint_freq // args.num_processes) == 0 and\
             (t + 1) not in full_checkpoints and args.save_checkpoints:
            trainer_save_name = f'{experiment_folder}/iter_{t}'
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
    exp_folder = log_dir / f'{args.exp_name}_{start_time}_{args.seed}'

    main(args, exp_folder)
