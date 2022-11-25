import gym
import numpy as np
import os
from collections import defaultdict
from schnetpack.data.loader import _collate_aseatoms

from rdkit.Chem import AllChem, MolFromSmiles, Conformer, AddHs

from .moldynamics_env import MolecularDynamics
from .xyz2mol import parse_molecule, get_rdkit_energy, set_coordinates
from .dft import atoms2psi4mol, get_dft_energy,\
     update_psi4_geometry, calculate_dft_energy_queue

RDKIT_ENERGY_THRESH = 500

class RewardWrapper(gym.Wrapper):
    molecules_xyz = {
        'C7O3C2OH8': 'aspirin.xyz',
        'N2C12H10': 'azobenzene.xyz',
        'C6H6': 'benzene.xyz',
        'C2OH6': 'ethanol.xyz',
        'C3O2H4': 'malonaldehyde.xyz',
        'C10H8': 'naphthalene.xyz',
        'C2ONC4OC2H9': 'paracetamol.xyz',
        'C3OC4O2H6': 'salicylic_acid.xyz',
        'C7H8': 'toluene.xyz',
        'C2NCNCO2H4': 'uracil.xyz'
    }

    def __init__(self,
                 env,
                 dft=False,
                 n_threads=1,
                 minimize_on_every_step=False,
                 greedy=False,
                 molecules_xyz_prefix='',
                 M=10,
                 done_when_not_improved=False):
        # Set arguments
        self.dft = dft
        self.n_threads = n_threads
        self.M = M
        self.minimize_on_every_step = minimize_on_every_step
        self.greedy = greedy
        self.molecules_xyz_prefix = molecules_xyz_prefix
        self.done_when_not_improved=done_when_not_improved

        self.update_coordinates = {
            'rdkit': set_coordinates,
            'dft': update_psi4_geometry
        }
        self.get_energy = {
            'rdkit': get_rdkit_energy,
            'dft': get_dft_energy
        }
        
        # Check parent class to name the reward correctly
        if isinstance(env, MolecularDynamics):
            self.reward_name = 'env_reward'
        else:
            self.reward_name = 'unknown_reward'
        super().__init__(env)
        
        # Initialize dictionaries
        self.n_parallel = self.env.n_parallel
        self.initial_energy = {
            'rdkit': [None] * self.n_parallel,
            'dft': [None] * self.n_parallel
        }
        self.molecule = {
            'rdkit': [None] * self.n_parallel,
            'dft': [None] * self.n_parallel
        }
        self.threshold_exceeded = [0.0 for _ in range(self.n_parallel)]
        self.molecules = {}
        self.parse_molecules()

    def parse_molecules(self):
        # Parse rdkit molecules 
        self.molecules['rdkit'] = {}
        for formula, path in RewardWrapper.molecules_xyz.items():
            molecule = parse_molecule(os.path.join(self.molecules_xyz_prefix, path))
            # Check if the provided molecule is valid
            try:
                self.get_energy['rdkit'](molecule)
            except AttributeError:
                raise ValueError("Provided molucule was not parsed correctly")
            self.molecules['rdkit'][formula] = molecule
    
    def step(self, actions):
        obs, env_rewards, dones, info = super().step(actions)
        
        # Put rewards from the environment into info
        info = dict(info, **{self.reward_name: env_rewards})

        # Get sizes of molecules
        atoms_num = self.env.get_atoms_num()
        env_steps = self.get_env_step()

        # Initialize reward arrays
        rewards = np.zeros(self.n_parallel)
        rdkit_rewards = np.zeros(self.n_parallel)
        final_energy = np.zeros(self.n_parallel)
        not_converged = np.zeros(self.n_parallel)
        threshold_exceeded_pct = np.zeros(self.n_parallel)

        # Initialize statistics for finished trajectories
        stats_done = defaultdict(lambda: [None] * self.n_parallel)

        # Rdkit rewards
        for idx in range(self.n_parallel):
            # Update current coordinates
            self.update_coordinates['rdkit'](self.molecule['rdkit'][idx], self.env.atoms[idx].get_positions())
            if self.dft:
                self.update_coordinates['dft'](self.molecule['dft'][idx], self.env.atoms[idx].get_positions())
            
            # Calculate current rdkit reward for every trajectory
            if self.minimize_on_every_step or info['env_done'][idx]:
                not_converged[idx], final_energy[idx] = self.minimize_rdkit(idx)
                rdkit_rewards[idx] = self.initial_energy['rdkit'][idx] - final_energy[idx]

        # DFT rewards
        if self.dft:
            queue = []
            for idx in range(self.n_parallel):
                if self.minimize_on_every_step or info['env_done'][idx]:
                    # Rdkit reward lower than RDKIT_DELTA_THRESH indicates highly improbable 
                    # conformations which are likely to cause an error in DFT calculation and/or
                    # significantly slow them down. To mitigate this we propose to replace DFT reward 
                    # in such states with rdkit reward. Note that rdkit reward is strongly 
                    # correlated with DFT reward and should not intefere with the training.
                    if final_energy[idx] < RDKIT_ENERGY_THRESH:
                        queue.append((self.molecule['dft'][idx], atoms_num[idx], idx))
                    else:
                        self.threshold_exceeded[idx] += 1
                        rewards[idx] = rdkit_rewards[idx]
            
            # Sort queue according to the molecule size
            queue = sorted(queue, key=lambda x:x[1], reverse=True)
            # TODO think about M=None, etc.
            result = calculate_dft_energy_queue(queue, n_threads=self.n_threads, M=self.M)
            for idx, _, energy in result:
                rewards[idx] = self.initial_energy['dft'][idx] - energy
        else:
            rewards = rdkit_rewards

        # Dones and info
        for idx in range(self.n_parallel):
            # If minimize_on_every step update initial energy
            if self.minimize_on_every_step:
                # initial_energy = final_energy
                self.initial_energy['rdkit'][idx] -= rdkit_rewards[idx]
                # FIXME ?
                # At the moment the wrapper is guaranteed to work correctly
                # only with done_when_not_improved=True. In case of greedy=True
                # we might get final_energy > RDKIT_ENERGY_THRESH on steps
                # [t, ..., t + T - 1] and then get final_energy > RDKIT_ENERGY_THRESH on
                # step t + T (although this is highly unlikely, it is possible).
                # Then the initial DFT energy would be calculated from the
                # rdkit reward but the final energy would come from DFT.
                if self.dft:
                    self.initial_energy['dft'][idx] -= rewards[idx]

            # If energy has not improved and done_when_not_improved=True set done to True 
            if self.done_when_not_improved and rewards[idx] < 0:
                dones[idx] = True

            # If TL is reached or done=True log final energy
            if dones[idx] or info['env_done'][idx] or self.greedy:
                # Increment number of finished trajectories3
                
                # Log final energy of the molecule
                stats_done['final_energy'][idx] = final_energy[idx]
                stats_done['not_converged'][idx] = not_converged[idx]
                
                # Log percentage of times in which treshold was  exceeded
                if self.dft:
                    threshold_exceeded_pct[idx] = self.threshold_exceeded[idx] / env_steps[idx]
                else:
                    threshold_exceeded_pct[idx] = 0
                stats_done['threshold_exceeded_pct'][idx] = threshold_exceeded_pct[idx]

                # Log final RL energy of the molecule
                if self.M > 0:
                    if self.dft:
                        self.update_coordinates['dft'](self.molecule['dft'][idx], self.env.atoms[idx].get_positions())
                        stats_done['final_rl_energy'][idx] = self.get_energy['dft'](self.molecule['dft'][idx])
                    else:
                        self.update_coordinates['rdkit'](self.molecule['rdkit'][idx], self.env.atoms[idx].get_positions())
                        stats_done['final_rl_energy'][idx] = self.get_energy['rdkit'](self.molecule['rdkit'][idx])
                else:
                    stats_done['final_rl_energy'][idx] = final_energy[idx]
        
        # Compute mean of stats over finished trajectories and update info
        info = dict(info, **stats_done)

        return obs, rewards, np.stack(dones), info
    
    def reset(self, indices=None):
        obs = self.env.reset(indices=indices)
        if indices is None:
            indices = np.arange(self.n_parallel)

        # Get sizes of molecules
        atoms_num = self.env.get_atoms_num()

        for idx in indices:
            # Calculate initial rdkit energy
            if self.env.smiles[idx] is not None:
                # Initialize molecule from Smiles
                self.molecule['rdkit'][idx] = MolFromSmiles(self.env.smiles[idx])
                self.molecule['rdkit'][idx] = AddHs(self.molecule['rdkit'][idx])
                # Add random conformer
                self.molecule['rdkit'][idx].AddConformer(Conformer(atoms_num[idx]))
            elif str(self.env.atoms[idx].symbols) in self.molecules['rdkit']:
                self.molecule['rdkit'][idx] = self.molecules['rdkit'][str(self.env.atoms[idx].symbols)]
            else:
                raise ValueError("Unknown molecule type {}".format(str(self.env.atoms[idx].symbols)))
            self.update_coordinates['rdkit'](self.molecule['rdkit'][idx], self.env.atoms[idx].get_positions())
            _, self.initial_energy['rdkit'][idx] = self.minimize_rdkit(idx)

            # Calculate initial dft energy
            if self.dft:
                queue = []
                self.threshold_exceeded[idx] = 0
                self.molecule['dft'][idx] = atoms2psi4mol(self.env.atoms[idx])
                if self.env.energy[idx] is not None and self.M == 0:
                    self.initial_energy['dft'][idx] = self.env.energy[idx]
                else:
                    queue.append((self.molecule['dft'][idx], atoms_num[idx], idx))
        
        # Calculate initial dft energy if it is not provided 
        if self.dft and len(queue) > 0:
            # Sort queue according to the molecule size
            queue = sorted(queue, key=lambda x:x[1], reverse=True)
            # TODO think about M=None, etc.
            result = calculate_dft_energy_queue(queue, n_threads=self.n_threads, M=self.M)
            for idx, _, energy in result:
                self.initial_energy['dft'][idx] = energy
    
        return obs

    def set_initial_positions(self, atoms_list, smiles_list, energy_list):
        super().reset()

        # Set molecules and get observation
        obs_list = []
        for idx, (atoms, smiles, energy) in enumerate(zip(atoms_list, smiles_list, energy_list)):
            self.env.atoms[idx] = atoms.copy()
            obs_list.append(self.env.converter(self.env.atoms[idx]))

            # Calculate initial rdkit energy
            if smiles is not None:
                # Initialize molecule from Smiles
                self.molecule['rdkit'][idx] = MolFromSmiles(smiles)
                self.molecule['rdkit'][idx] = AddHs(self.molecule['rdkit'][idx])
                # Add random conformer
                self.molecule['rdkit'][idx].AddConformer(Conformer(len(atoms.get_atomic_numbers())))
            elif str(self.env.atoms[idx].symbols) in self.molecules['rdkit']:
                self.molecule['rdkit'][idx] = self.molecules['rdkit'][str(self.env.atoms[idx].symbols)]
            else:
                raise ValueError("Unknown molecule type {}".format(str(self.env.atoms[idx].symbols)))
            self.update_coordinates['rdkit'](self.molecule['rdkit'][idx], self.env.atoms[idx].get_positions())
            _, self.initial_energy['rdkit'][idx] = self.minimize_rdkit(idx)

            # Calculate initial dft energy
            if self.dft:
                queue = []
                self.threshold_exceeded[idx] = 0
                self.molecule['dft'][idx] = atoms2psi4mol(self.env.atoms[idx])
                if energy is not None and self.M == 0:
                    self.initial_energy['dft'][idx] = energy
                else:
                    queue.append((self.molecule['dft'][idx], len(atoms.get_atomic_numbers()), idx))
        
        # Calculate initial dft energy if it is not provided 
        if self.dft and len(queue) > 0:
            # Sort queue according to the molecule size
            queue = sorted(queue, key=lambda x:x[1], reverse=True)
            # TODO think about M=None, etc.
            result = calculate_dft_energy_queue(queue, n_threads=self.n_threads, M=self.M)
            for idx, _, energy in result:
                self.initial_energy['dft'][idx] = energy


        obs = _collate_aseatoms(obs_list)
        obs = {k:v.squeeze(1) for k, v in obs.items()}
        return obs

    def minimize_rdkit(self, idx, M=None):
        # Set number of minization iterations
        if M is None:
            n_its = self.M
        else:
            n_its = M

        # Perform rdkit minimization
        ff = AllChem.MMFFGetMoleculeForceField(self.molecule['rdkit'][idx],
                AllChem.MMFFGetMoleculeProperties(self.molecule['rdkit'][idx]), confId=0)
        ff.Initialize()
        not_converged = ff.Minimize(maxIts=n_its)
        energy = self.get_energy['rdkit'](self.molecule['rdkit'][idx])
       
        return not_converged, energy

    def get_atoms_num(self):
        return self.env.get_atoms_num()

    def get_env_step(self):
        return self.env.get_env_step()

    def update_timelimit(self, tl):
        return self.env.update_timelimit(tl)
