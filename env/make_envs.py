from .optimization_env import OptimizationEnv
from .wrappers import EnergyWrapper


def make_envs(args, neural_oracle):
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
        "surrogate_oracle": args.surrogate_oracle_type,
        "tau": args.tau,
        "neural_oracle": neural_oracle,
        "n_threads": args.n_threads,
        "minimize_on_every_step": args.minimize_on_every_step,
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
    reward_wrapper_kwargs["minimize_on_every_step"] = False

    # Initialize eval env
    eval_env = OptimizationEnv(**env_kwargs)
    eval_env = EnergyWrapper(eval_env, **reward_wrapper_kwargs)

    return env, eval_env
