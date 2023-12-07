from .optimization_env import OptimizationEnv
from .wrappers import EnergyWrapper


def make_envs(args):
    # Env kwargs
    env_kwargs = {
        "db_path": args.db_path,
        "n_parallel": args.n_parallel,
        "timelimit": args.timelimit_train,
        "sample_initial_conformations": True,
        "num_initial_conformations": args.num_initial_conformations,
    }

    # Reward wrapper kwargs
    reward_wrapper_kwargs = {
        "dft": args.reward == "dft",
        "n_threads": args.n_threads,
        "minimize_on_every_step": args.minimize_on_every_step,
        "evaluation": False,
        "terminate_on_negative_reward": args.terminate_on_negative_reward,
        "max_num_negative_rewards": args.max_num_negative_rewards,
        "host_file_path": args.host_file_path,
    }

    # Initialize env
    env = OptimizationEnv(**env_kwargs)
    env = EnergyWrapper(env, **reward_wrapper_kwargs)

    # Update kwargs for eval_env
    if args.eval_db_path != "":
        env_kwargs["db_path"] = args.eval_db_path
    env_kwargs.update(
        {
            "sample_initial_conformations": args.sample_initial_conformations,
            "timelimit": args.timelimit_eval,
        }
    )

    if args.reward == "rdkit":
        env_kwargs["n_parallel"] = 1
    else:
        env_kwargs["n_parallel"] = args.n_eval_runs
        reward_wrapper_kwargs["minimize_on_every_step"] = False
        reward_wrapper_kwargs["evaluation"] = True

    # Initialize eval env
    eval_env = OptimizationEnv(**env_kwargs)
    eval_env = EnergyWrapper(eval_env, **reward_wrapper_kwargs)

    return env, eval_env
