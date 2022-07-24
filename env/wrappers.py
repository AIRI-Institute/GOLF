import gym
import os

from rdkit.Chem import AllChem, rdmolops

from .moldynamics_env import MolecularDynamics
from .xyz2mol import parse_molecule, get_rdkit_energy, set_coordinates


class RdkitMinimizationReward(gym.Wrapper):
    molecules_xyz = {
        'C7O3C2OH8': 'env/molecules_xyz/aspirin.xyz',
        'N2C12H10': 'env/molecules_xyz/azobenzene.xyz',
        'C6H6': 'env/molecules_xyz/benzene.xyz',
        'C2OH6': 'env/molecules_xyz/ethanol.xyz',
        'C3O2H4': 'env/molecules_xyz/malonaldehyde.xyz',
        'C10H8': 'env/molecules_xyz/naphthalene.xyz',
        'C2ONC4OC2H9': 'env/molecules_xyz/paracetamol.xyz',
        'C3OC4O2H6': 'env/molecules_xyz/salicylic_acid.xyz',
        'C7H8': 'env/molecules_xyz/toluene.xyz',
        'C2NCNCO2H4': 'env/molecules_xyz/uracil.xyz'
    }

    def __init__(self,
                 env,
                 minimize_on_every_step=False,
                 remove_hydrogen=False,
                 M=10):
        # Initialize molecule's sructure
        self.M = M
        self.minimize_on_every_step = minimize_on_every_step
        self.remove_hydrogen = remove_hydrogen
        self.initial_energy = 0
        # Parse molecules
        self.molecules = {}
        for formula, path in RdkitMinimizationReward.molecules_xyz.items():

            molecule = parse_molecule(os.path.join(os.getcwd(), path))
            # Check if the provided molecule is valid
            try:
                get_rdkit_energy(molecule)
            except AttributeError:
                raise ValueError("Provided molucule was not parsed correctly")
            if self.remove_hydrogen:
                # Remove hydrogen atoms from the molecule
                self.molecule = rdmolops.RemoveHs(molecule)
            self.molecules[formula] = molecule
        self.molecule = None

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
            not_converged, final_energy = self.minimize()
            reward = self.initial_energy - final_energy
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
        self.molecule = self.molecules[str(self.env.atoms.symbols)]
        set_coordinates(self.molecule, self.env.atoms.get_positions())
        # Minimize the initial state of the molecule
        _, self.initial_energy = self.minimize()
        return obs

    def set_initial_positions(self, atoms):
        self.env.reset()
        self.env.atoms = atoms.copy()

        self.molecule = self.molecules[str(self.env.atoms.symbols)]
        set_coordinates(self.molecule, self.env.atoms.get_positions())
        # Minimize the initial state of the molecule
        _, self.initial_energy = self.minimize()

    def minimize(self, M=None, confId=0):
        # Set number of minization iterations
        if M is None:
            n_its = self.M
        else:
            n_its = M    
        if self.remove_hydrogen:
            # Add hydrogens back to the molecule
            self.molecule = rdmolops.AddHs(self.molecule, addCoords=True)
        ff = AllChem.MMFFGetMoleculeForceField(self.molecule,
                AllChem.MMFFGetMoleculeProperties(self.molecule), confId=confId)
        ff.Initialize()
        not_converged = ff.Minimize(maxIts=n_its)
        energy = get_rdkit_energy(self.molecule)
        if self.remove_hydrogen:
            # Remove hydrogens after minimization
            self.molecule = rdmolops.RemoveHs(self.molecule)
        return not_converged, energy


def rdkit_reward_wrapper(**kwargs):
    env = RdkitMinimizationReward(**kwargs)
    return env