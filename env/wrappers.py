import gym
import os
import psi4

from abc import abstractmethod

from rdkit.Chem import AllChem, rdmolops
from psi4.driver.p4util.exceptions import OptimizationConvergenceError

from .moldynamics_env import MolecularDynamics
from .xyz2mol import parse_molecule, get_rdkit_energy, set_coordinates
from .dft import parse_psi4_molecule, get_dft_energy, update_psi4_geometry

class BaseRewardWrapper(gym.Wrapper):
    molecules_xyz = {
        #'C7O3C2OH8': 'env/molecules_xyz/aspirin.xyz',
        #'N2C12H10': 'env/molecules_xyz/azobenzene.xyz',
        #'C6H6': 'env/molecules_xyz/benzene.xyz',
        #'C2OH6': 'env/molecules_xyz/ethanol.xyz',
        'C3O2H4': 'env/molecules_xyz/malonaldehyde.xyz',
        #'C10H8': 'env/molecules_xyz/naphthalene.xyz',
        #'C2ONC4OC2H9': 'env/molecules_xyz/paracetamol.xyz',
        #'C3OC4O2H6': 'env/molecules_xyz/salicylic_acid.xyz',
        #'C7H8': 'env/molecules_xyz/toluene.xyz',
        #'C2NCNCO2H4': 'env/molecules_xyz/uracil.xyz'
    }

    def __init__(self,
                 env,
                 minimize_on_every_step=False,
                 greedy=False,
                 remove_hydrogen=False,
                 molecules_xyz_prefix='',
                 M=10,
                 done_when_not_improved=False):
        # Set arguments
        self.M = M
        self.minimize_on_every_step = minimize_on_every_step
        self.greedy = greedy
        self.remove_hydrogen = remove_hydrogen
        self.molecules_xyz_prefix = molecules_xyz_prefix
        self.done_when_not_improved=done_when_not_improved

        self.initial_energy = 0
        self.parse_molecule = None
        self.update_coordinates = None
        self.get_energy = None
        
        # Check parent class to name the reward correctly
        if isinstance(env, MolecularDynamics):
            self.reward_name = 'env_reward'
        else:
            self.reward_name = 'unknown_reward'
        super().__init__(env)

    def parse_molecules(self):
        self.molecules = {}
        for formula, path in BaseRewardWrapper.molecules_xyz.items():
            molecule = self.parse_molecule(os.path.join(self.molecules_xyz_prefix, path))
            # Check if the provided molecule is valid
            try:
                self.get_energy(molecule)
            except AttributeError:
                raise ValueError("Provided molucule was not parsed correctly")
            if self.remove_hydrogen:
                # Remove hydrogen atoms from the molecule
                self.molecule = rdmolops.RemoveHs(molecule)
            self.molecules[formula] = molecule
        self.molecule = None

    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        info = dict(info, **{self.reward_name: reward})
        
        # Update current coordinates
        self.update_coordinates(self.molecule, self.env.atoms.get_positions())
        
        # Minimize with rdkit and calculate reward
        if self.minimize_on_every_step or info['env_done']:
            not_converged, final_energy, rdkit_final_energy = self.minimize()
            info['rdkit_final_energy'] = rdkit_final_energy
            info['dft_final_energy'] = final_energy
            info['dft_exception'] = int(not not_converged)
            reward = self.initial_energy - final_energy
        else:
            reward = 0.

        # If minimize_on_every step update initial energy
        if self.minimize_on_every_step:
            self.initial_energy = final_energy

        # If energy has not improved and done_when_not_improved=True set done to True 
        if self.done_when_not_improved and reward < 0:
            done = True

        # If TL is reached or done=True log final energy
        if done or info['env_done'] or self.greedy:
            info['final_energy'] = final_energy
            info['not_converged'] = not_converged
            if self.M > 0:
                self.update_coordinates(self.molecule, self.env.atoms.get_positions())
                if self.remove_hydrogen:
                    # Add hydrogens back to the molecule to evaluate final rl energy
                    self.molecule = rdmolops.AddHs(self.molecule, addCoords=True)
                info['final_rl_energy'] = self.get_energy(self.molecule)
            else:
                info['final_rl_energy'] = final_energy
            if self.remove_hydrogen:
                # Remove hydrogens after energy calculation
                self.molecule = rdmolops.RemoveHs(self.molecule)
        
        return obs, reward, done, info
    
    def reset(self):
        obs = super().reset()
        self.molecule = self.molecules[str(self.env.atoms.symbols)]
        self.update_coordinates(self.molecule, self.env.atoms.get_positions())
        # Minimize the initial state of the molecule
        _, self.initial_energy, _ = self.minimize()

        return obs

    def set_initial_positions(self, atoms, M=None):
        self.env.reset()
        self.env.atoms = atoms.copy()
        obs = self.env.converter(self.env.atoms)

        self.molecule = self.molecules[str(self.env.atoms.symbols)]
        self.update_coordinates(self.molecule, self.env.atoms.get_positions())
        # Minimize the initial state of the molecule
        _, self.initial_energy, self.rdkit_energy = self.minimize(M)

        return obs

    @abstractmethod
    def minimize(self, M=None, confId=0):
        raise NotImplementedError()

    def get_atoms_num(self):
        return self.env.get_atoms_num()

    def get_env_step(self):
        return self.env.get_env_step()

    def update_timelimit(self, tl):
        return self.env.update_timelimit(tl)


class RdkitMinimizationReward(BaseRewardWrapper):
    def __init__(self,
                 env,
                 minimize_on_every_step=False,
                 greedy=False,
                 remove_hydrogen=False,
                 molecules_xyz_prefix='',
                 M=10,
                 done_when_not_improved=False):
        super().__init__(env, minimize_on_every_step, greedy, remove_hydrogen,
                         molecules_xyz_prefix, M, done_when_not_improved)

        # Rdkit-specific functions
        self.parse_molecule = parse_molecule
        self.get_energy = get_rdkit_energy
        self.update_coordinates = set_coordinates
        # Parse molecules
        self.parse_molecules()
    
    def minimize(self, M=None):
        # Set number of minization iterations
        if M is None:
            n_its = self.M
        else:
            n_its = M    
        if self.remove_hydrogen:
            # Add hydrogens back to the molecule
            self.molecule = rdmolops.AddHs(self.molecule, addCoords=True)
        ff = AllChem.MMFFGetMoleculeForceField(self.molecule,
                AllChem.MMFFGetMoleculeProperties(self.molecule), confId=0)
        ff.Initialize()
        not_converged = ff.Minimize(maxIts=n_its)
        energy = get_rdkit_energy(self.molecule)
        if self.remove_hydrogen:
            # Remove hydrogens after minimization
            self.molecule = rdmolops.RemoveHs(self.molecule)
        return not_converged, energy


class DFTMinimizationReward(BaseRewardWrapper):
    def __init__(self,
                 env,
                 minimize_on_every_step=False,
                 greedy=False,
                 remove_hydrogen=False,
                 molecules_xyz_prefix='',
                 M=10,
                 done_when_not_improved=False):
        super().__init__(env, minimize_on_every_step, greedy, remove_hydrogen,
                         molecules_xyz_prefix, M, done_when_not_improved)

        assert not remove_hydrogen, "'remove_hydrogen = True' is not supported for DFT!"

        # DFT-specific functions
        self.parse_molecule = parse_psi4_molecule
        self.get_energy = get_dft_energy
        self.update_coordinates = update_psi4_geometry
        # Parse molecules
        self.parse_molecules()
    
    def minimize(self, M=None):
        # Set number of minization iterations
        if M is None:
            n_its = self.M
        else:
            n_its = M
        not_converged = True
        rdkit_energy = 0.0
        if n_its > 0:
            psi4.set_options({'geom_maxiter': n_its})
            try:
                energy = psi4.optimize("wb97x-d/def2-svp", **{"molecule": self.molecule, "return_wfn": False})
                not_converged = False
            except OptimizationConvergenceError as e:
                self.molecule.set_geometry(e.wfn.molecule().geometry())
                energy = e.wfn.energy()
            psi4.core.clean()
        else:
            # Calculate rdkit energy
            rdkit_molecules = {}
            for formula, path in BaseRewardWrapper.molecules_xyz.items():
                rdkit_molecule = parse_molecule(os.path.join(self.molecules_xyz_prefix, path))
                rdkit_molecules[formula] = rdkit_molecule
            rdkit_molecule = rdkit_molecules[str(self.env.atoms.symbols)]
            set_coordinates(rdkit_molecule,  self.env.atoms.get_positions())
            rdkit_energy = get_rdkit_energy(rdkit_molecule)
            energy = self.get_energy(self.molecule)
            # Dirty hack to detect SCFConvergenceError will be removed
            if energy == -200.0:
                not_converged = False

        return not_converged, energy, rdkit_energy


def reward_wrapper(reward, **kwargs):
    if reward == "rdkit":
        env = RdkitMinimizationReward(**kwargs)
    elif reward == "dft":
        env = DFTMinimizationReward(**kwargs)
    else:
        raise NotImplementedError()
    return env
