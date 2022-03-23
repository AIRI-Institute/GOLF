
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


def env(**kwargs):
    '''
    The env function often wraps the environment in wrappers by default.
    '''
    env = raw_env(**kwargs)
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(**kwargs):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = MolecularDynamics(**kwargs)
    env = parallel_to_aec(env)
    return env

class MolecularDynamics(ParallelEnv):
    metadata = {"render_modes":["human"], "name":"md_v0"}
    atomic_numbers_mapping = {
        1: "hydrogen",
        6: "carbon",
        7: "nitrogen",
        8: "oxygen"
    }    
    
    def __init__(self, db_path, timelimit=1000, exp_folder=None,
                 save_trajectories=False, evaluate_rewards_rdkit=False) -> None:
        self.TL = timelimit
        self.dbpath = db_path
        self.db_len = self._get_db_length()
        self.exp_folder = exp_folder
        self.save_trajectories = save_trajectories
        self.evaluate_rewards_rdkit = evaluate_rewards_rdkit
        self.atoms = None

        assert self.exp_folder is not None, "Provide a name for the experiment in order to save trajectories."
        self.traj_file = os.path.join(exp_folder, 'tmp.traj')

        with connect(self.dbpath) as conn:
            example_atoms = conn.get(1).toatoms()
        self.atomic_numbers = example_atoms.get_atomic_numbers()
        observation_shape = example_atoms.get_positions().shape

        self.possible_agents, self.action_spaces = self._init_agents()
        self.observation_spaces = [
            Box(
                low=-np.inf, 
                high=np.inf,
                shape=(observation_shape)
            )
            for _ in self.possible_agents
        ]

    def _init_agents(self):
        unit_cube_action_space = Box(low=-1.0, high=1.0, shape=(3,))
        last_type = ""
        agents = []
        self.agent_name_mapping = {}
        action_spaces = {}
        i = 0
        for agent_id, atomic_number in enumerate(self.atomic_numbers):
            if atomic_number not in self.atomic_numbers_mapping:
                raise ValueError("Invalid atomic number for atom number {}".format(agent_id))
            agent_type = self.atomic_numbers_mapping[atomic_number]
            if last_type == agent_type:
                i += 1
            else:
                i = 0
            agents.append(f"{agent_type}_{i}")
            self.agent_name_mapping[agents[-1]] = agent_id
            action_spaces[agents[-1]] = unit_cube_action_space
            last_type = agent_type
        return agents, action_spaces

    def step(self, actions) -> dict:
        current_postitions = self.atoms.get_positions()
        
        actions_np = np.zeros(current_postitions.shape)
        for agent in self.agents:
            actions_np[self.agent_name_mapping[agent], :] = actions[agent]
        print(actions_np)
        
        current_postitions += actions_np
        print(current_postitions)
        self.atoms.set_positions(current_postitions)
        print(self.atoms.get_positions())
        self.trajectory.write(self.atoms)

        self.env_steps += 1
        env_done = self.env_steps >= self.TL

        if self.evaluate_rewards_rdkit:
            rewards = self._evaluate_rewards_rdkit()
        else:
            rewards = {agent: None for agent in self.agents}                

        observations = {agent: self.atoms for agent in self.agents}
        dones = {agent: env_done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, dones, infos

    def render(self, mode="human") -> None:
        if self.env_done:
            trajectory = Trajectory(self.traj_file, "r")
            view(trajectory, viewer='nglviewer')
            if self.save_trajectory:
                new_traj_file = os.path.join(self.exp_folder, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.traj')
                os.replace(self.traj_file, new_traj_file)
            os.remove(self.traj_file)
    
    def reset(self, idx=None) -> dict:
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
        observations = {agent: self.atoms for agent in self.agents}
        return observations

    def close(self) -> None:
        if os.path.exists(self.traj_file):
            os.remove(self.traj_file)

    def _evaluate_rewards_rdkit(self):
        pass

    def _get_db_length(self):
        with connect(self.dbpath) as conn:
            db_len = len(conn)
        return db_len
