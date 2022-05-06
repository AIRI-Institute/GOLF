import gym
import torch

from .moldynamics_env import MolecularDynamics
from .xyz2mol import parse_molecule, get_rdkit_energy, set_coordinates


class ma_gym_schnet_reward(gym.Wrapper):
    def __init__(self, env, model, reward_delta=False):
        self.model = model
        self.reward_delta = reward_delta
        self.prev_reward = 0
        # Check parent class to name the reward correctly
        if isinstance(env, ma_gym_rdkit_reward):
            self.reward_name = 'rdkit_reward'
        elif isinstance(env, MolecularDynamics):
            self.reward_name = 'env_reward'
        else:
            self.reward_name = 'unknown_reward'
        self.mean_energy = env.mean_energy
        self.std_energy = env.std_energy
        # Reward delta can only be used without normalization
        if self.reward_delta:
            assert self.mean_energy == 0. and self.std_energy == 1., \
                "Set calculate_mean_std_energy to False to use reward delta"
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
        rewards = {agent: self.prev_reward - (schnet_output['energy'].item() - self.mean_energy) / self.std_energy for agent in obs.keys()}
        if self.reward_delta:
            self.prev_reward = schnet_output['energy'].item()
        return obs, rewards, done, infos

    def reset(self):
        obs = super().reset()
        if self.reward_delta:
            self.prev_reward = self.model(obs)
        return obs


class gym_schnet_reward(gym.Wrapper):
    def __init__(self, env, model, reward_delta=False):
        self.model = model
        self.reward_delta = reward_delta
        self.prev_reward = 0
        # Check parent class to name the reward correctly
        if isinstance(env, gym_rdkit_reward):
            self.reward_name = 'rdkit_reward'
        elif isinstance(env, MolecularDynamics):
            self.reward_name = 'env_reward'
        else:
            self.reward_name = 'unknown_reward'
        self.mean_energy = env.mean_energy
        self.std_energy = env.std_energy
        # Reward delta can only be used without normalization
        if self.reward_delta:
            assert self.mean_energy == 0. and self.std_energy == 1., \
                "Set calculate_mean_std_energy to False to use reward delta"
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # Save environment rewards in infos
        info = dict(info, **{self.reward_name: reward})
        # Calculate Schnet rewards
        schnet_output = self.model(obs)
        # Return calculated rewards

        reward = self.prev_reward - (schnet_output['energy'].item() - self.mean_energy) / self.std_energy
        if self.reward_delta:
            self.prev_reward = schnet_output['energy'].item()
        return obs, reward, done, info
    
    def reset(self):
        obs = super().reset()
        if self.reward_delta:
            self.prev_reward = self.model(obs)['energy'].item()
        return obs


class ma_gym_rdkit_reward(gym.Wrapper):
    def __init__(self, env, molecule_path, reward_delta=False):
        # Initialize molecule's sructure
        self.reward_delta = reward_delta
        self.prev_reward = 0
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
        rewards = {agent: self.prev_reward - rdkit_output for agent in obs.keys()}
        if self.reward_delta:
            self.prev_reward = rdkit_output
        return obs, rewards, done, infos

    def reset(self):
        obs = super().reset()
        if self.reward_delta:
            set_coordinates(self.molecule, self.env.atoms.get_positions())
            self.prev_reward = get_rdkit_energy(self.molecule)
        return obs


class gym_rdkit_reward(gym.Wrapper):
    def __init__(self, env, molecule_path, reward_delta=False):
        # Initialize molecule's sructure
        self.reward_delta = reward_delta
        self.prev_reward = 0
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
        reward = self.prev_reward - rdkit_output
        if self.reward_delta:
            self.prev_reward = rdkit_output
        return obs, reward, done, info
    
    def reset(self):
        obs = super().reset()
        if self.reward_delta:
            set_coordinates(self.molecule, self.env.atoms.get_positions())
            self.prev_reward = get_rdkit_energy(self.molecule)
        return obs


def schnet_reward_wrapper(env, multiagent, schnet_model_path, reward_delta, device):
    model = torch.load(schnet_model_path, map_location=device)
    if multiagent:
        env = ma_gym_schnet_reward(env, model, reward_delta)
    else:
        env = gym_schnet_reward(env, model, reward_delta)
    return env

def rdkit_reward_wrapper(env, multiagent, molecule_path, reward_delta):
    if multiagent:
        env = ma_gym_rdkit_reward(env, molecule_path, reward_delta)
    else:
        env = gym_rdkit_reward(env, molecule_path, reward_delta)
    return env