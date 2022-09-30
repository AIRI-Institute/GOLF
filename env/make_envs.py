from .subproc_env import SubprocVecEnv
from .moldynamics_env import env_fn
from .wrappers import rdkit_reward_wrapper
from rl import DEVICE

def make_env(env_kwargs, reward_wrapper_kwargs, seed, rank):
    def _thunk():
        env = env_fn(DEVICE, **env_kwargs)
        env.seed(seed + rank)
        env = rdkit_reward_wrapper(env=env, **reward_wrapper_kwargs)
        return env
    return _thunk

def make_envs(env_kwargs, reward_wrapper_kwargs, seed, num_processes):
    envs = [make_env(env_kwargs, reward_wrapper_kwargs, seed, i) for i in range(num_processes)]
    envs = SubprocVecEnv(envs)
    return envs