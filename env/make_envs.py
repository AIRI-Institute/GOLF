from .moldynamics_env import env_fn
from .wrappers import RewardWrapper
 
def make_envs(args):
    # Env kwargs
    env_kwargs = {
        'db_path': args.db_path,
        'n_parallel': args.n_parallel,
        'timelimit': args.timelimit,
        'done_on_timelimit': args.done_on_timelimit,
        'sample_initial_conformations': True,
        'num_initial_conformations': args.num_initial_conformations,
    }
    
    # Reward wrapper kwargs
    reward_wrapper_kwargs = {
        'dft': args.reward == 'dft',
        'n_threads': args.n_threads,
        'minimize_on_every_step': args.minimize_on_every_step,
        'greedy': args.greedy,
        'molecules_xyz_prefix': args.molecules_xyz_prefix,
        'M': args.M,
        'done_when_not_improved': args.done_when_not_improved
    }

    # Initialize env
    env = env_fn(**env_kwargs)
    env = RewardWrapper(env, **reward_wrapper_kwargs)
    
    # Update kwargs for eval_env
    if args.eval_db_path != '':
        env_kwargs['db_path'] = args.eval_db_path

    # Set timelimit to 100 to correctly log eval/episode_len
    env_kwargs['timelimit'] = 100
    if args.reward == "rdkit":
        env_kwargs['n_parallel'] = 1
    else:
        env_kwargs['n_parallel'] = args.n_eval_runs
        reward_wrapper_kwargs['n_threads'] = args.n_eval_runs
    
    # Initialize eval env
    eval_env = env_fn(**env_kwargs)
    eval_env = RewardWrapper(eval_env, **reward_wrapper_kwargs)

    return env, eval_env