import gym
import numpy as np
import torch
from torch_geometric.data import Batch, Data

from env.dft_worker import update_ase_atoms_positions
from env.oracles import DFTOracle, NeuralOracle, RdkitOracle
from env.xyz2mol import set_coordinates
from GOLF import DEVICE
from utils.utils import ignore_extra_args

RDKIT_ORACLE_THRESH = 300
# Neural oracle predicts formation energies
# Formation energy > 0.0 indicates an unrealistic conformation
NEURAL_ORACLE_THRESH = 0.0
KCALMOL2HARTREE = 627.5

surrogate_oracles = {
    "rdkit": ignore_extra_args(RdkitOracle),
    "neural": ignore_extra_args(NeuralOracle),
}


class EnergyWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        dft=False,
        n_threads=1,
        surrogate_oracle="rdkit",
        tau=0.5,
        neural_oracle=None,
        minimize_on_every_step=False,
        minimize_on_done=True,
        terminate_on_negative_reward=False,
        max_num_negative_rewards=1,
        host_file_path=None,
    ):
        # Set arguments
        self.dft = dft
        self.n_threads = n_threads
        self.minimize_on_every_step = minimize_on_every_step
        self.minimize_on_done = minimize_on_done
        self.terminate_on_negative_reward = terminate_on_negative_reward
        self.max_num_negative_rewards = max_num_negative_rewards

        if surrogate_oracle == "rdkit":
            self.thresh = RDKIT_ORACLE_THRESH
        elif surrogate_oracle == "neural":
            self.thresh = NEURAL_ORACLE_THRESH

        # Initialize environemnt
        super().__init__(env)
        self.n_parallel = self.env.n_parallel

        # Initialize surrogate oracle
        surrogate_oracle_args = {
            "n_parallel": self.n_parallel,
            "update_coordinates_fn": set_coordinates,
            "model": neural_oracle,
            "tau": tau,
        }
        self.surrogate_oracle = surrogate_oracles[surrogate_oracle](
            **surrogate_oracle_args
        )

        # Initialize DFT oracle
        self.genuine_oracle = DFTOracle(
            n_parallel=self.n_parallel,
            update_coordinates_fn=update_ase_atoms_positions,
            n_threads=self.n_threads,
            host_file_path=host_file_path,
        )

        self.negative_rewards_counter = np.zeros(self.n_parallel)

    def step(self, actions):
        obs, env_rewards, dones, info = super().step(actions)
        dones = np.stack(dones)

        # Put rewards from the environment into info
        info = dict(info, **{"env_reward": env_rewards})

        # Rdkit rewards
        new_positions = [molecule.get_positions() for molecule in self.env.atoms]
        self.surrogate_oracle.update_coordinates(new_positions)
        if self.dft:
            self.genuine_oracle.update_coordinates(new_positions)

        # Calculate rdkit energies and forces
        calc_SO_energy_indices = np.where(
            self.minimize_on_every_step | (self.minimize_on_done & dones)
        )[0]

        new_energies, forces = self.surrogate_oracle.calculate_energies_forces(
            indices=calc_SO_energy_indices
        )

        # Update current energies and forces. Calculate delta energy
        self.surrogate_oracle.update_forces(forces, calc_SO_energy_indices)
        energy_delta = self.surrogate_oracle.get_energy_delta(
            new_energies, calc_SO_energy_indices
        )

        # When agent encounters 'max_num_negative_rewards' terminate the episode
        self.negative_rewards_counter[energy_delta < 0] += 1
        dones[self.negative_rewards_counter >= self.max_num_negative_rewards] = True

        # Log final energies of molecules
        info["final_energy"] = self.surrogate_oracle.initial_energies

        # DFT energies
        if self.dft:
            # Conformations whose energy w.r.t. to surrogate  is higher than
            # THRESHOLD are highly improbable and likely to cause
            # an error in DFT calculation and/or significantly
            # slow them down. To mitigate this we propose to replace the DFT reward
            # in such states with the Rdkit reward, as they are strongly correlated in such states.
            SO_energy_thresh_exceeded = (
                self.surrogate_oracle.initial_energies >= self.thresh
            )

            # Calculate energy and forces with DFT only for terminal states.
            # Skip conformations with energy higher than RDKIT_ENERGY_THRESH
            calculate_dft_energy_env_ids = np.where(
                self.minimize_on_done & dones & ~SO_energy_thresh_exceeded
            )[0]
            if len(calculate_dft_energy_env_ids) > 0:
                info["calculate_dft_energy_env_ids"] = calculate_dft_energy_env_ids
            self.genuine_oracle.submit_tasks(calculate_dft_energy_env_ids)

        return obs, energy_delta, dones, info

    def reset(self, indices=None):
        obs = self.env.reset(indices=indices)
        if indices is None:
            indices = np.arange(self.n_parallel)

        # Reset negative rewards counter
        self.negative_rewards_counter[indices] = 0

        # Get sizes of molecules
        smiles_list = [self.env.smiles[i] for i in indices]
        molecules = [self.env.atoms[i].copy() for i in indices]
        self.surrogate_oracle.initialize_molecules(indices, smiles_list, molecules)

        if self.dft:
            dft_initial_energies = [self.env.energy[i] for i in indices]
            dft_forces = [self.env.force[i] for i in indices]
            self.genuine_oracle.initialize_molecules(
                indices, molecules, dft_initial_energies, dft_forces
            )

        return obs

    def set_initial_positions(
        self, molecules, smiles_list, energy_list, force_list, max_its=0
    ):
        super().reset(increment_conf_idx=False)
        indices = np.arange(self.n_parallel)

        # Reset negative rewards counter
        self.negative_rewards_counter.fill(0.0)

        obs_list = []
        # Set molecules and get observation
        for i, molecule in enumerate(molecules):
            self.env.atoms[i] = molecule.copy()
            obs_list.append(
                Data(
                    z=torch.from_numpy(molecule.get_atomic_numbers()).long(),
                    pos=torch.from_numpy(molecule.get_positions()).float(),
                ).to(DEVICE)
            )

        self.surrogate_oracle.initialize_molecules(
            indices, smiles_list, molecules, max_its
        )

        if self.dft:
            self.genuine_oracle.initialize_molecules(
                indices, molecules, energy_list, force_list
            )

        obs = Batch.from_data_list(obs_list)
        return obs

    def update_timelimit(self, tl):
        return self.env.update_timelimit(tl)

    def get_forces(self, indices=None):
        if self.dft:
            return self.genuine_oracle.get_forces(indices)
        else:
            return self.surrogate_oracle.get_forces(indices)

    def get_energies(self, indices=None):
        if self.dft:
            return self.genuine_oracle.get_energies(indices)
        else:
            return self.surrogate_oracle.get_energies(indices)

    def save_surrogate_oracle(self, path):
        self.surrogate_oracle.save(path)
