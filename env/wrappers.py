import gym
import os
import psi4

from rdkit.Chem import AllChem, rdmolops
from psi4.driver.p4util.exceptions import OptimizationConvergenceError

from .moldynamics_env import MolecularDynamics
from .xyz2mol import parse_molecule, get_rdkit_energy, set_coordinates
from .dft import parse_psi4_molecule, get_dft_energy, update_psi4_geometry, FUNCTIONAL_STRING

RDKIT_ENERGY_THRESH = 200

class RewardWrapper(gym.Wrapper):
    molecules_xyz = {
        # 'C7O3C2OH8': 'env/molecules_xyz/aspirin.xyz',
        # 'N2C12H10': 'env/molecules_xyz/azobenzene.xyz',
        # 'C6H6': 'env/molecules_xyz/benzene.xyz',
        # 'C2OH6': 'env/molecules_xyz/ethanol.xyz',
        'C3O2H4': 'env/molecules_xyz/malonaldehyde.xyz',
        # 'C10H8': 'env/molecules_xyz/naphthalene.xyz',
        # 'C2ONC4OC2H9': 'env/molecules_xyz/paracetamol.xyz',
        # 'C3OC4O2H6': 'env/molecules_xyz/salicylic_acid.xyz',
        # 'C7H8': 'env/molecules_xyz/toluene.xyz',
        # 'C2NCNCO2H4': 'env/molecules_xyz/uracil.xyz'
    }

    def __init__(self,
                 env,
                 dft=False,
                 minimize_on_every_step=False,
                 greedy=False,
                 remove_hydrogen=False,
                 molecules_xyz_prefix='',
                 M=10,
                 done_when_not_improved=False):
        # Set arguments
        self.dft = dft
        self.M = M
        self.minimize_on_every_step = minimize_on_every_step
        self.greedy = greedy
        self.remove_hydrogen = remove_hydrogen
        self.molecules_xyz_prefix = molecules_xyz_prefix
        self.done_when_not_improved=done_when_not_improved

        self.initial_energy = {}
        self.parse_molecule = {
            'rdkit': parse_molecule,
            'dft': parse_psi4_molecule
        }
        self.update_coordinates = {
            'rdkit': set_coordinates,
            'dft': update_psi4_geometry
        }
        self.get_energy = {
            'rdkit': get_rdkit_energy,
            'dft': get_dft_energy
        }

        self.molecules = {}
        self.molecule = {}
        self.parse_molecules()
        
        # Check parent class to name the reward correctly
        if isinstance(env, MolecularDynamics):
            self.reward_name = 'env_reward'
        else:
            self.reward_name = 'unknown_reward'
        super().__init__(env)

    def parse_molecules(self):
        # Parse rdkit molecules 
        self.molecules['rdkit'] = {}
        for formula, path in RewardWrapper.molecules_xyz.items():
            molecule = self.parse_molecule['rdkit'](os.path.join(self.molecules_xyz_prefix, path))
            # Check if the provided molecule is valid
            try:
                self.get_energy['rdkit'](molecule)
            except AttributeError:
                raise ValueError("Provided molucule was not parsed correctly")
            if self.remove_hydrogen:
                # Remove hydrogen atoms from the molecule
                molecule = rdmolops.RemoveHs(molecule)
            self.molecules['rdkit'][formula] = molecule
        self.molecule['rdkit'] = None

        # Parse DFT molecules if needed
        if self.dft:
            self.molecules['dft'] = {}
            for formula, path in RewardWrapper.molecules_xyz.items():
                molecule = self.parse_molecule['dft'](os.path.join(self.molecules_xyz_prefix, path))
                # Check if the provided molecule is valid
                try:
                    self.get_energy['dft'](molecule)
                except AttributeError:
                    raise ValueError("Provided molucule was not parsed correctly")
                self.molecules['dft'][formula] = molecule
            self.molecule['dft'] = None

    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        info = dict(info, **{self.reward_name: reward})
        
        # Update current coordinates
        self.update_coordinates['rdkit'](self.molecule['rdkit'], self.env.atoms.get_positions())
        if self.dft:
            self.update_coordinates['dft'](self.molecule['dft'], self.env.atoms.get_positions())
        
        # Calculate reward
        if self.minimize_on_every_step or info['env_done']:
            not_converged, final_energy = self.minimize_rdkit()
            rdkit_reward = self.initial_energy['rdkit'] - final_energy
            # Rdkit reward lower than RDKIT_DELTA_THRESH indicates highly improbable 
            # conformations which are likely to cause an error in DFT calculation and/or
            # significantly slow them down. To mitigate this we propose to replace DFT reward 
            # in such states with rdkit reward. Note that rdkit reward is strongly 
            # correlated with DFT reward and should not intefere with the training.
            if not self.dft or (self.dft and final_energy > RDKIT_ENERGY_THRESH):
                if self.dft:
                    self.threshold_exceeded += 1
                reward = rdkit_reward
            else:
                not_converged, final_energy = self.minimize_dft()
                reward = self.initial_energy['dft'] - final_energy
        else:
            reward = 0.

        # If minimize_on_every step update initial energy
        if self.minimize_on_every_step:
            # initial_energy = final_energy
            self.initial_energy['rdkit'] -= rdkit_reward
            # FIXME ?
            # At the moment the wrapper is guaranteed to work correctly
            # only with done_when_not_improved=True. In case of greedy=True
            # we might get final_energy > RDKIT_ENERGY_THRESH on steps
            # [t, ..., t + T - 1] and then get final_energy > RDKIT_ENERGY_THRESH on
            # step t + T (although this is highly unlikely, it is possible).
            # Then the initial DFT energy would be calculated from the
            # rdkit reward but the final energy would come from DFT.
            if self.dft:
                self.initial_energy['dft'] -= reward

        # If energy has not improved and done_when_not_improved=True set done to True 
        if self.done_when_not_improved and reward < 0:
            done = True

        # If TL is reached or done=True log final energy
        if done or info['env_done'] or self.greedy:
            # Log final energy of the molecule
            info['final_energy'] = final_energy
            info['not_converged'] = not_converged
            
            # Log percentage of times in which treshold was  exceeded
            if self.dft:
                threshold_exceeded_pct = self.threshold_exceeded / self.get_env_step()
            else:
                threshold_exceeded_pct = 0
            info['threshold_exceeded_pct'] = threshold_exceeded_pct

            # Log final RL energy of the molecule
            if self.M > 0:
                if self.dft:
                    self.update_coordinates['dft'](self.molecule['dft'], self.env.atoms.get_positions())
                    info['final_rl_energy'] = self.get_energy['dft'](self.molecule['dft'])
                else:
                    self.update_coordinates['rdkit'](self.molecule['rdkit'], self.env.atoms.get_positions())
                    if self.remove_hydrogen:
                        # Add hydrogens back to the molecule to evaluate final rl energy
                        self.molecule['rdkit'] = rdmolops.AddHs(self.molecule['rdkit'], addCoords=True)
                    info['final_rl_energy'] = self.get_energy(self.molecule['rdkit'])
                    if self.remove_hydrogen:
                        # Remove hydrogens after energy calculation
                        self.molecule['rdkit'] = rdmolops.RemoveHs(self.molecule['rdkit'])
            else:
                info['final_rl_energy'] = final_energy
        
        return obs, reward, done, info
    
    def reset(self):
        obs = super().reset()
        
        # Calculate initial rdkit energy
        self.molecule['rdkit'] = self.molecules['rdkit'][str(self.env.atoms.symbols)]
        self.update_coordinates['rdkit'](self.molecule['rdkit'], self.env.atoms.get_positions())
        _, self.initial_energy['rdkit'] = self.minimize_rdkit()

        # Calculate initial dft energy
        if self.dft:
            self.threshold_exceeded = 0
            self.molecule['dft'] = self.molecules['dft'][str(self.env.atoms.symbols)]
            self.update_coordinates['dft'](self.molecule['dft'], self.env.atoms.get_positions())
            _,  self.initial_energy['dft'] = self.minimize_dft()
        return obs

    def set_initial_positions(self, atoms, M=None):
        self.env.reset()
        self.env.atoms = atoms.copy()
        obs = self.env.converter(self.env.atoms)

        # Calculate initial rdkit energy
        self.molecule['rdkit'] = self.molecules['rdkit'][str(self.env.atoms.symbols)]
        self.update_coordinates['rdkit'](self.molecule['rdkit'], self.env.atoms.get_positions())
        _, self.initial_energy['rdkit'] = self.minimize_rdkit()

        # Calculate initial dft energy
        if self.dft:
            self.threshold_exceeded = 0
            self.molecule['dft'] = self.molecules['dft'][str(self.env.atoms.symbols)]
            self.update_coordinates['dft'](self.molecule['dft'], self.env.atoms.get_positions())
            _,  self.initial_energy['dft'] = self.minimize_dft()
        return obs

    def minimize_rdkit(self, M=None, confId=0):
        # Set number of minization iterations
        if M is None:
            n_its = self.M
        else:
            n_its = M

        # Perform rdkit minimization
        if self.remove_hydrogen:
            # Add hydrogens back to the molecule
            self.molecule['rdkit'] = rdmolops.AddHs(self.molecule['rdkit'], addCoords=True)
        ff = AllChem.MMFFGetMoleculeForceField(self.molecule['rdkit'],
                AllChem.MMFFGetMoleculeProperties(self.molecule['rdkit']), confId=0)
        ff.Initialize()
        not_converged = ff.Minimize(maxIts=n_its)
        energy = self.get_energy['rdkit'](self.molecule['rdkit'])
        if self.remove_hydrogen:
            # Remove hydrogens after minimization
            self.molecule['rdkit'] = rdmolops.RemoveHs(self.molecule['rdkit'])
        
        return not_converged, energy

    def minimize_dft(self, M=None):
        # Set number of minization iterations
        if M is None:
            n_its = self.M
        else:
            n_its = M
        
        # Perform DFT minimization
        not_converged = True
        if n_its > 0:
            psi4.set_options({'geom_maxiter': n_its})
            try:
                energy = psi4.optimize(FUNCTIONAL_STRING, **{"molecule": self.molecule['dft'], "return_wfn": False})
                not_converged = False
            except OptimizationConvergenceError as e:
                self.molecule['dft'].set_geometry(e.wfn.molecule().geometry())
                energy = e.wfn.energy()
            # Hartree to kcal/mol
            energy *= 627.5
            psi4.core.clean()
        else:
            # Calculate DFT energy
            energy = self.get_energy['dft'](self.molecule['dft'])
        
        return not_converged, energy

    def get_atoms_num(self):
        return self.env.get_atoms_num()

    def get_env_step(self):
        return self.env.get_env_step()

    def update_timelimit(self, tl):
        return self.env.update_timelimit(tl)
