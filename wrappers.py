import gym
import torch

from ase.atoms import Atoms
from schnetpack.data.atoms import AtomsConverter


class gym_schnet_reward(gym.Wrapper):
    def __init__(self, env, model, converter):
        self.converter = converter
        self.model = model
        super().__init__(env)

    def step(self, action):
        obs, rewards, done, infos = super().step(action)
        # Save environment rewards in infos
        infos = {agent: dict(infos[agent], env_reward=rewards[agent]) \
                 for agent in obs.keys()}
        # Calculate Schnet rewards
        # We assume that atoms can observe the states of other atoms
        # Thus all agents share the same observation
        atoms = list(obs.values())[0]
        assert isinstance(atoms, Atoms), "Observation must be an Atoms object"
        obs_schnet = self.converter(atoms)
        schnet_output = self.model(obs_schnet)
        # Return calculated rewards
        rewards = {agent: schnet_output['energy'].item() for agent in obs.keys()}
        return obs, rewards, done, infos

def schnet_reward_wrapper(env, schnet_model_path, device):
    model = torch.load(schnet_model_path, map_location=torch.device('cpu'))
    converter = AtomsConverter(device=device)
    env = gym_schnet_reward(env, model, converter)
    return env