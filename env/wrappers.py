import gym
import torch

from rdkit.Chem import AllChem

from .moldynamics_env import MolecularDynamics
from .xyz2mol import parse_molecule, get_rdkit_energy, set_coordinates


class rdkit_minization_reward(gym.Wrapper):
    def __init__(self, env, molecule_path, minimize_on_every_step=False, M=10):
        # Initialize molecule's sructure
        self.M = M
        self.minimize_on_every_step = minimize_on_every_step
        self.initial_energy = 0
        molecule = parse_molecule(molecule_path)

        # Check if the provided molecule is valid
        try:
            get_rdkit_energy(molecule)
        except AttributeError:
            raise ValueError("Provided molucule was not parsed correctly")
        self.molecule = molecule

        # Check parent class to name the reward correctly
        if isinstance(env, MolecularDynamics):
            self.reward_name = 'env_reward'
        else:
            self.reward_name = 'unknown_reward'
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        info = dict(info, **{self.reward_name: reward})
        
        # Update current coordinates
        set_coordinates(self.molecule, self.env.atoms.get_positions())
        
        # Minimize with rdkit and calculate reward
        if self.minimize_on_every_step or info['env_done']:
            ff = AllChem.MMFFGetMoleculeForceField(self.molecule,
                    AllChem.MMFFGetMoleculeProperties(self.molecule), confId=0)
            ff.Initialize()
            not_converged = ff.Minimize(maxIts=self.M)
            # Get energy after minimization
            final_energy = get_rdkit_energy(self.molecule)
            reward = self.initial_energy - final_energy
        else:
            reward = 0.

        # If TL is reached log final energy
        if info['env_done']:
            info['final_energy'] = final_energy
            info['not_converged'] = not_converged
            set_coordinates(self.molecule, self.env.atoms.get_positions())
            info['final_rl_energy'] = get_rdkit_energy(self.molecule)
        
        return obs, reward, done, info
    
    def reset(self):
        obs = super().reset()
        set_coordinates(self.molecule, self.env.atoms.get_positions())
        self.initial_energy = get_rdkit_energy(self.molecule)
        return obs

def rdkit_reward_wrapper(env, molecule_path, minimize_on_every_step, M):
    env = rdkit_minization_reward(env, molecule_path, minimize_on_every_step, M)
    return env