import gym
import torch

from rdkit.Chem import AllChem, rdmolops

from .moldynamics_env import MolecularDynamics
from .xyz2mol import parse_molecule, get_rdkit_energy, set_coordinates


class rdkit_minization_reward(gym.Wrapper):
    def __init__(self,
                 env,
                 molecule_path,
                 minimize_on_every_step=False,
                 remove_hydrogen=False,
                 M=10):
        # Initialize molecule's sructure
        self.M = M
        self.minimize_on_every_step = minimize_on_every_step
        self.remove_hydrogen = remove_hydrogen
        self.initial_energy = 0
        molecule = parse_molecule(molecule_path)

        # Check if the provided molecule is valid
        try:
            get_rdkit_energy(molecule)
        except AttributeError:
            raise ValueError("Provided molucule was not parsed correctly")
        self.molecule = molecule
        if self.remove_hydrogen:
            # Remove hydrogen atoms from the molecule
            self.molecule = rdmolops.RemoveHs(molecule)

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
            not_converged = self.minimize(self.remove_hydrogen)
            # Get energy after minimization
            final_energy = get_rdkit_energy(self.molecule)
            reward = self.initial_energy - final_energy
            if self.remove_hydrogen:
                # Remove hydrogens after minimization
                self.molecule = rdmolops.RemoveHs(self.molecule)
        else:
            reward = 0.

        # If minimize_on_every step update initial energy
        if self.minimize_on_every_step:
            self.initial_energy = final_energy

        # If TL is reached log final energy
        if info['env_done']:
            info['final_energy'] = final_energy
            info['not_converged'] = not_converged
            set_coordinates(self.molecule, self.env.atoms.get_positions())
            if self.remove_hydrogen:
                # Add hydrogens back to the molecule to evaluate final rl energy
                self.molecule = rdmolops.AddHs(self.molecule, addCoords=True)
            info['final_rl_energy'] = get_rdkit_energy(self.molecule)
            if self.remove_hydrogen:
                # Remove hydrogens after energy calculation
                self.molecule = rdmolops.RemoveHs(self.molecule)
        
        return obs, reward, done, info
    
    def reset(self):
        obs = super().reset()
        set_coordinates(self.molecule, self.env.atoms.get_positions())
        # Minimize the initial state of the molecule
        self.minimize(self.remove_hydrogen)
        self.initial_energy = get_rdkit_energy(self.molecule)
        if self.remove_hydrogen:
            # Remove hydrogens after minimization
            self.molecule = rdmolops.RemoveHs(self.molecule)
        return obs

    def minimize(self, remove_hydrogen, M=None, confId=0):
        # Set number of minization iterations
        if M is None:
            n_its = self.M
        else:
            n_its = M    
        if remove_hydrogen:
            # Add hydrogens back to the molecule
            self.molecule = rdmolops.AddHs(self.molecule, addCoords=True)
        ff = AllChem.MMFFGetMoleculeForceField(self.molecule,
                AllChem.MMFFGetMoleculeProperties(self.molecule), confId=confId)
        ff.Initialize()
        not_converged = ff.Minimize(maxIts=n_its)
        return not_converged

def rdkit_reward_wrapper(**kwargs):
    env = rdkit_minization_reward(**kwargs)
    return env