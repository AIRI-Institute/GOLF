import backoff
import gym
import numpy as np
import os

from datetime import datetime

from ase.db import connect
from ase.io import Trajectory
from ase.visualize import view

from gym.spaces import Box, Dict
from sqlite3 import DatabaseError
from schnetpack.data.atoms import AtomsConverter

from .ma_moldynamics import MAMolecularDynamics


def on_giveup(details):
    print("Giving Up after {} tries. Time elapsed: {:.3f} :(".format(details['tries'], details['elapsed']))


class MolecularDynamics(gym.Env):
    metadata = {"render_modes":["human"], "name":"md_v0"}
    
    def __init__(self,
                 db_path,
                 converter, 
                 timelimit=100,
                 calculate_mean_std=False,
                 done_on_timelimit=False,
                 inject_noise=False,
                 noise_std=0.1,
                 exp_folder=None,
                 save_trajectories=False):
        self.TL = timelimit
        self.done_on_timelimit = done_on_timelimit
        self.inject_noise = inject_noise
        self.noise_std = noise_std
        self.dbpath = db_path
        self.converter = converter
        self.db_len = self._get_db_length()
        self.exp_folder = exp_folder
        self.save_trajectories = save_trajectories
        self.env_done = True
        self.atoms = None
        self.mean_energy = 0.
        self.std_energy = 1.
        if calculate_mean_std:
            self.mean_energy, self.std_energy = self._get_mean_std_energy()
        self.initial_molecule_conformations = []
        
        assert self.exp_folder is not None, "Provide a name for the experiment in order to save trajectories."
        self.traj_file = os.path.join(exp_folder, 'tmp.traj')

        # Store random subset of molecules DB
        self._get_initial_molecule_conformations()
        example_row = self.initial_molecule_conformations[0]
        # Initialize observaition space
        self.atoms_count = sum(example_row.count_atoms().values())
        self.example_atoms = example_row.toatoms()
        self.observation_space = Dict(
            {
                '_atomic_numbers': Box(low=0.0, high=9.0, shape=self.example_atoms.get_atomic_numbers().shape, dtype=np.int64),
                '_positions': Box(low=-np.inf, high=np.inf, shape=self.example_atoms.get_positions().shape, dtype=np.float32),
                '_neighbors': Box(low=0.0, high=self.atoms_count - 1, shape=(self.atoms_count, self.atoms_count - 1), dtype=np.int64),
                '_cell': Box(low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float32),
                '_cell_offset': Box(low=-np.inf, high=np.inf, shape=(self.atoms_count, self.atoms_count - 1, 3), dtype=np.float32),
                '_atom_mask': Box(low=0.0, high=1.0, shape=(self.atoms_count, ), dtype=np.float32),
                '_neighbor_mask': Box(low=0.0, high=1.0, shape=(self.atoms_count, self.atoms_count - 1), dtype=np.float32)
            }
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.atoms_count, 3), dtype=np.float32)

    def step(self, actions):
        current_postitions = self.atoms.get_positions()
        current_postitions += actions
        self.atoms.set_positions(current_postitions)
        self.trajectory.write(self.atoms)

        self.env_steps += 1
        self.env_done = self.env_steps >= self.TL

        reward = None
        obs = self.converter(self.atoms)
        if self.done_on_timelimit:
            done = self.env_done
        else:
            done = False
        info = {'env_done': self.env_done}

        return obs, reward, done, info

    def render(self, mode="human"):
        if self.env_done:
            trajectory = Trajectory(self.traj_file, "r")
            view(trajectory)
            if self.save_trajectories:
                new_traj_file = os.path.join(self.exp_folder, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.traj')
                os.replace(self.traj_file, new_traj_file)
            else:
                os.remove(self.traj_file)
    
    def reset(self, db_idx=None):
        if db_idx is None:
            db_idx = np.random.randint(len(self.initial_molecule_conformations))
        self.atoms = self.initial_molecule_conformations[db_idx].toatoms()
        # Inject noise into the initial state 
        # to make optimal initital states less optimal
        if self.inject_noise:
            current_positions = self.atoms.get_positions()
            noise = np.random.normal(scale=self.noise_std, size=current_positions.shape)
            self.atoms.set_positions(current_positions + noise)
        if os.path.exists(self.traj_file):
            os.remove(self.traj_file)
        self.trajectory = Trajectory(self.traj_file, 'w')
        self.trajectory.write(self.atoms)

        self.env_steps = 0
        self.env_done = False
        obs = self.converter(self.atoms)
        return obs

    def close(self):
        if os.path.exists(self.traj_file):
            os.remove(self.traj_file)

    def _get_db_length(self):
        with connect(self.dbpath) as conn:
            db_len = len(conn)
        return db_len

    def _get_initial_molecule_conformations(self):
        # 50000 is a randomly chosen constant. Should be enough
        random_sample_size = min(self.db_len, 50000)
        self.initial_molecule_conformations = []
        indices = np.random.choice(np.arange(1, self.db_len + 1), random_sample_size, replace=False)
        for idx in indices:
            self.initial_molecule_conformations.append(self._get_molecule(int(idx)))
    
    # Makes sqllite3 database compatible with NFS storages
    @backoff.on_exception(
        backoff.expo,
        exception=DatabaseError,
        max_tries=5,
        on_giveup=on_giveup
    )
    def _get_molecule(self, idx):
        with connect(self.dbpath) as conn:
            return conn.get(idx)

    def _get_mean_std_energy(self):
        energy = []
        # Speed up the computation
        indices = np.random.choice(np.arange(1, self.db_len + 1), self.db_len // 10, replace=False)
        for idx in indices:
            row = self._get_molecule(int(idx))
            energy.append(row.data['energy'])
        energy = np.array(energy)
        return energy.mean(), energy.std()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        return seed

def env_fn(device, multiagent=False, **kwargs):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    converter = AtomsConverter(device=device)
    if multiagent is True:
        env = MAMolecularDynamics(converter=converter, **kwargs)
    else:
        env = MolecularDynamics(converter=converter, **kwargs)
    return env
