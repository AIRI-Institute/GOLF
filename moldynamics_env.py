
import os
import numpy as np

from datetime import datetime

from ase.db import connect
from ase.io import Trajectory
from ase.visualize import view

from gym.spaces import Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec


# def make_env(raw_env):
#     '''
#     The env function often wraps the environment in wrappers by default.
#     '''
#     def env_fn(**kwargs):
#         env = raw_env(**kwargs)
#         # This wrapper is only for environments which print results to the terminal
#         env = wrappers.CaptureStdoutWrapper(env)
#         # Provides a wide vareity of helpful user errors
#         # Strongly recommended
#         env = wrappers.OrderEnforcingWrapper(env)
#         return env
#     return env_fn

def env_fn(**kwargs):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = MolecularDynamics(**kwargs)
    # env = wrappers.OrderEnforcingWrapper(env)
    return env

class MolecularDynamics(ParallelEnv):
    metadata = {"render_modes":["human"], "name":"md_v0"}
    
    def __init__(self, db_path, timelimit=1000, exp_folder=None,
                 save_trajectories=False):
        self.TL = timelimit
        self.dbpath = db_path
        self.db_len = self._get_db_length()
        self.exp_folder = exp_folder
        self.save_trajectories = save_trajectories
        self.env_done = True
        self.atoms = None

        assert self.exp_folder is not None, "Provide a name for the experiment in order to save trajectories."
        self.traj_file = os.path.join(exp_folder, 'tmp.traj')

        with connect(self.dbpath) as conn:
            example_row = conn.get(1)
        self.atoms_count = example_row.count_atoms()
        example_atoms = example_row.toatoms()
        self.observation_shape = example_atoms.get_positions().shape

        self.possible_agents, self.action_spaces, self.observation_spaces = self._init_agents()

    def _init_agents(self):
        unit_cube_action_space = Box(low=-1.0, high=1.0, shape=(3,))
        r3_observation_space = Box(low=-np.inf, high=np.inf, shape=self.observation_shape)

        agents = []
        self.agent_name_mapping = {}
        action_spaces = {}
        observation_spaces = {}

        for atom_type, atom_count in self.atoms_count.items():
            for atom_number in range(atom_count):
                agent_id = len(agents)
                agents.append(f"{atom_type}_{atom_number}")
                self.agent_name_mapping[agents[-1]] = agent_id
                action_spaces[agents[-1]] = unit_cube_action_space
                observation_spaces[agents[-1]] = r3_observation_space
        return agents, action_spaces, observation_spaces

    def step(self, actions):
        #if self.env_done:
        #    raise AssertionError("reset() needs to be called before step")
        current_postitions = self.atoms.get_positions()
        
        actions_np = np.zeros(current_postitions.shape)
        for agent in self.agents:
            actions_np[self.agent_name_mapping[agent], :] = actions[agent]
        
        current_postitions += actions_np
        self.atoms.set_positions(current_postitions)
        self.trajectory.write(self.atoms)

        self.env_steps += 1
        self.env_done = self.env_steps >= self.TL

        rewards = {agent: None for agent in self.agents}                
        observations = {agent: self.atoms for agent in self.agents}
        dones = {agent: self.env_done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self.agents = [
            agent for agent in self.agents if not dones[agent]
        ]

        return observations, rewards, dones, infos

    def render(self, mode="human"):
        if self.env_done:
            trajectory = Trajectory(self.traj_file, "r")
            view(trajectory)
            if self.save_trajectories:
                new_traj_file = os.path.join(self.exp_folder, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.traj')
                os.replace(self.traj_file, new_traj_file)
            else:
                os.remove(self.traj_file)
    
    def reset(self, idx=None):
        if idx is None:
            idx = np.random.randint(1, self.db_len + 1)
        with connect(self.dbpath) as conn:
            atoms = conn.get(idx).toatoms()
        self.atoms = atoms
        
        if os.path.exists(self.traj_file):
            os.remove(self.traj_file)
        self.trajectory = Trajectory(self.traj_file, 'w')
        self.trajectory.write(self.atoms)

        self.agents = self.possible_agents[:]
        self.env_steps = 0
        self.env_done = False
        observations = {agent: self.atoms for agent in self.agents}
        return observations

    def close(self):
        if os.path.exists(self.traj_file):
            os.remove(self.traj_file)

    def _get_db_length(self):
        with connect(self.dbpath) as conn:
            db_len = len(conn)
        return db_len
