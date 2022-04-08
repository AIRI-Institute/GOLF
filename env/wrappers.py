import gym
import torch

from .moldynamics_env import MolecularDynamics
from .xyz2mol import parse_molecule, get_rdkit_energy, set_coordinates


class ma_gym_schnet_reward(gym.Wrapper):
    def __init__(self, env, model):
        self.model = model
        # Check parent class to name the reward correctly
        if isinstance(env, ma_gym_rdkit_reward):
            self.reward_name = 'rdkit_reward'
        elif isinstance(env, MolecularDynamics):
            self.reward_name = 'env_reward'
        else:
            self.reward_name = 'unknown_reward'
        super().__init__(env)

    def step(self, action):
        obs, rewards, done, infos = super().step(action)
        # Save environment rewards in infos
        infos = {agent: dict(infos[agent], **{self.reward_name: rewards[agent]}) \
                 for agent in obs.keys()}
        # Calculate Schnet rewards
        # We assume that atoms can observe the states of other atoms
        # Thus all agents share the same observation
        schnet_output = self.model(list(obs.values())[0])
        # Return calculated rewards
        rewards = {agent: schnet_output['energy'].item() for agent in obs.keys()}
        return obs, rewards, done, infos


class gym_schnet_reward(gym.Wrapper):
    def __init__(self, env, model):
        self.model = model
        # Check parent class to name the reward correctly
        if isinstance(env, gym_rdkit_reward):
            self.reward_name = 'rdkit_reward'
        elif isinstance(env, MolecularDynamics):
            self.reward_name = 'env_reward'
        else:
            self.reward_name = 'unknown_reward'
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # Save environment rewards in infos
        info = dict(info, **{self.reward_name: reward})
        # Calculate Schnet rewards
        # We assume that atoms can observe the states of other atoms
        # Thus all agents share the same observation
        schnet_output = self.model(obs)
        # Return calculated rewards
        reward = schnet_output['energy'].item()
        return obs, reward, done, info


class ma_gym_rdkit_reward(gym.Wrapper):
    def __init__(self, env, molecule_path):
        # Initialize molecule's sructure
        molecule = parse_molecule(molecule_path)
        try:
            get_rdkit_energy(molecule)
        except AttributeError:
            raise ValueError("Provided molucule was not parsed correctly")
        self.molecule = molecule
        # Check parent class to name the reward correctly
        if isinstance(env, ma_gym_schnet_reward):
            self.reward_name = 'schnet_reward'
        elif isinstance(env, MolecularDynamics):
            self.reward_name = 'env_reward'
        else:
            self.reward_name = 'unknown_reward'
        super().__init__(env)
    
    def step(self, action):
        obs, rewards, done, infos = super().step(action)
        infos = {agent: dict(infos[agent], **{self.reward_name: rewards[agent]}) \
                 for agent in obs.keys()}
        # Update current coordinates
        set_coordinates(self.molecule, self.env.atoms.get_positions())
        rdkit_output = get_rdkit_energy(self.molecule)
        rewards = {agent: rdkit_output for agent in obs.keys()}
        return obs, rewards, done, infos


class gym_rdkit_reward(gym.Wrapper):
    def __init__(self, env, molecule_path):
        # Initialize molecule's sructure
        molecule = parse_molecule(molecule_path)
        try:
            get_rdkit_energy(molecule)
        except AttributeError:
            raise ValueError("Provided molucule was not parsed correctly")
        self.molecule = molecule
        # Check parent class to name the reward correctly
        if isinstance(env, gym_schnet_reward):
            self.reward_name = 'schnet_reward'
        elif isinstance(env, MolecularDynamics):
            self.reward_name = 'env_reward'
        else:
            self.reward_name = 'unknown_reward'
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        info = dict(info, **{self.reward_name: reward})
        # Update current coordinates
        set_coordinates(self.molecule, self.env.atoms.get_positions())
        rdkit_output = get_rdkit_energy(self.molecule)
        return obs, rdkit_output, done, info


def schnet_reward_wrapper(env, multiagent, schnet_model_path, device):
    model = torch.load(schnet_model_path, map_location=device)
    if multiagent:
        env = ma_gym_schnet_reward(env, model)
    else:
        env = gym_schnet_reward(env, model)
    return env

def rdkit_reward_wrapper(env, multiagent, molecule_path):
    if multiagent:
        env = ma_gym_rdkit_reward(env, molecule_path)
    else:
        env = gym_rdkit_reward(env, molecule_path)
    return env
