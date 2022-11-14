import backoff
import gym
import numpy as np
import torch
import warnings

from ase.db import connect
from sqlite3 import DatabaseError
from schnetpack.data.atoms import AtomsConverter
from schnetpack.data.loader import _collate_aseatoms


np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# For backoff exceptions
def on_giveup(details):
    print("Giving Up after {} tries. Time elapsed: {:.3f} :(".format(details['tries'], details['elapsed']))


class MolecularDynamics(gym.Env):
    metadata = {"render_modes":["human"], "name":"md_v0"}
    
    def __init__(self,
                 db_path,
                 converter,
                 n_parallel=1,
                 timelimit=10,
                 done_on_timelimit=False,
                 sample_initial_conformations=True,
                 num_initial_conformations=50000,
                ):
        self.db_path = db_path
        self.converter = converter
        self.n_parallel=n_parallel
        self.TL = timelimit
        self.done_on_timelimit = done_on_timelimit
        self.sample_initial_conformations = sample_initial_conformations
        
        self.db_len = self.get_db_length()
        self.env_done = True
        self.atoms = None
        self.mean_energy = 0.
        self.std_energy = 1.
        self.initial_molecule_conformations = []

        # Store random subset of molecules DB
        self.get_initial_molecule_conformations(num_initial_conformations)
        self.conformation_idx = 0

        # Initialize lists
        self.atoms = [None] * self.n_parallel
        self.smiles = [None] * self.n_parallel
        self.energy = [None] * self.n_parallel
        self.env_steps = [None] * self.n_parallel
        self.env_done = [None] * self.n_parallel

    def step(self, actions):
        # Get number of atoms in each molecule
        numbers_atoms = self.get_atoms_num()

        obs = []
        rewards = [None] * self.n_parallel
        dones = [None] * self.n_parallel
        info = {'env_done': [None] * self.n_parallel}

        for idx, (number_atoms, action) in enumerate(zip(numbers_atoms, actions)):
            # Unpad action
            self.atoms[idx].set_positions(self.atoms[idx].get_positions() + action[:number_atoms])

            self.env_steps[idx] += 1
            self.env_done[idx] = self.env_steps[idx] >= self.TL

            # Convert atoms to obs
            obs.append(self.converter(self.atoms[idx]))
            
            if self.done_on_timelimit:
                dones[idx] = self.env_done[idx]
            else:
                dones[idx] = False
            info['env_done'][idx] = self.env_done[idx]
        
        # Collate observations into a batch
        obs = _collate_aseatoms(obs)
        obs = {k:v.squeeze(1) for k, v in obs.items()}

        return obs, rewards, dones, info

    def reset(self, indices=None):
        # If indices is not provided reset all molecules
        if indices is None:
            indices = np.arange(self.n_parallel)

        # If sample_initial_conformations iterate over all initial conformations sequentially
        if self.sample_initial_conformations:
            db_indices = np.random.choice(len(self.initial_molecule_conformations), len(indices), replace=False)
        else:
            start_conf_idx = self.conformation_idx % len(self.initial_molecule_conformations)
            db_indices = np.arange(start_conf_idx, start_conf_idx + len(indices))
            self.conformation_idx += len(indices)

        rows = [self.initial_molecule_conformations[db_idx] for db_idx in db_indices]

        obs = []
        for idx, row in zip(indices, rows):
            # Copy to avoid changing the atoms object inplace
            self.atoms[idx] = row.toatoms().copy()

            # Check if row has Smiles
            if hasattr(row, 'smiles'):
                self.smiles[idx] = row.smiles
                self.energy[idx] = row.data['energy']

            # Reset env_steps and done
            self.env_steps[idx] = 0
            self.env_done[idx] = False

            # Convert atoms to obs
            obs.append(self.converter(self.atoms[idx]))

        # Collate observations into a batch
        obs = _collate_aseatoms(obs)
        obs = {k:v.squeeze(1) for k, v in obs.items()}

        return obs

    def update_timelimit(self, new_timelimit):
        self.TL = new_timelimit

    def get_db_length(self):
        with connect(self.db_path) as conn:
            db_len = len(conn)
        return db_len

    def get_env_step(self):
        return self.env_steps

    def get_initial_molecule_conformations(self, num_initial_conformations):
        if num_initial_conformations == -1 or num_initial_conformations == self.db_len:
            indices = np.arange(1, self.db_len + 1)
        else:
            indices = np.random.choice(np.arange(1, self.db_len + 1), min(self.db_len, num_initial_conformations), replace=False)
        self.initial_molecule_conformations = []
        for idx in indices:
            row = self.get_molecule(int(idx))
            self.initial_molecule_conformations.append(row)
    
    # Makes sqllite3 database compatible with NFS storages
    @backoff.on_exception(
        backoff.expo,
        exception=DatabaseError,
        jitter=backoff.full_jitter,
        max_tries=10,
        on_giveup=on_giveup
    )
    def get_molecule(self, idx):
        with connect(self.db_path) as conn:
            return conn.get(idx)

    def get_atoms_num(self):
        return [len(atom.get_atomic_numbers()) for atom in self.atoms]

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        return seed

def env_fn(**kwargs):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    converter = AtomsConverter(device=torch.device('cpu'))
    env = MolecularDynamics(converter=converter, **kwargs)
    return env
