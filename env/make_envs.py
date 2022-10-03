from .moldynamics_env import env_fn
from .subproc_env import SubprocVecEnv
from .wrappers import reward_wrapper
from rl import DEVICE

def make_env(env_kwargs, reward_type, reward_wrapper_kwargs, seed, rank):
    def _thunk():
        env = env_fn(**env_kwargs)
        env.seed(seed + rank)
        env = reward_wrapper(reward_type, env=env, **reward_wrapper_kwargs)
        return env
    return _thunk

def make_envs(env_kwargs, reward_type, reward_wrapper_kwargs, seed, num_processes):
    envs = [make_env(env_kwargs, reward_type, reward_wrapper_kwargs, seed, i) for i in range(num_processes)]
    envs = SubprocVecEnv(DEVICE, envs)
    return envs