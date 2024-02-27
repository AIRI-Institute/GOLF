import gym
import concurrent.futures
import numpy as np
import math
import multiprocessing as mp
import torch

from rdkit.Chem import AddHs, AllChem, Conformer, MolFromSmiles, SmilesParserParams
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.interfaces import AtomsConverter
from schnetpack.transform import ASENeighborList

from .dft_worker import update_ase_atoms_positions
from .dft import get_dft_server_destinations, calculate_dft_energy_tcp_client
from .xyz2mol import get_rdkit_energy, get_rdkit_force, set_coordinates

RDKIT_ENERGY_THRESH = 300
KCALMOL2HARTREE = 627.5


class BaseOracle:
    def __init__(self, n_parallel, update_coordinates_fn):
        self.n_parallel = n_parallel
        self.update_coordinates_fn = update_coordinates_fn

        self.initial_energies = np.zeros(self.n_parallel)
        self.forces = [None] * self.n_parallel
        self.molecules = [None] * self.n_parallel

    def get_energies(self, indices):
        if indices is None:
            indices = np.arange(self.n_parallel)
        return self.initial_energies[indices]

    def get_forces(self, indices):
        if indices is None:
            indices = np.arange(self.n_parallel)
        return [np.array(self.forces[i]) for i in indices]

    def update_coordinates(self, positions, indices=None):
        if indices is None:
            indices = np.arange(self.n_parallel)
        assert len(positions) == len(
            indices
        ), f"Not enough values to update all molecules! Expected {self.n_parallel} but got {len(positions)}"

        # Update current molecules
        for i, position in zip(indices, positions):
            self.update_coordinates_fn(self.molecules[i], position)

    def update_forces(self, forces, indices=None):
        if indices is None:
            indices = np.arange(self.n_parallel)
        assert len(forces) == len(indices)
        for i, force in zip(indices, forces):
            self.forces[i] = force


class RdkitOracle(BaseOracle):
    def __init__(self, n_parallel, update_coordinates_fn):
        super().__init__(n_parallel, update_coordinates_fn)

    def calculate_energies_forces(self, max_its=0, indices=None):
        if indices is None:
            indices = np.arange(self.n_parallel)

        not_converged, energies, forces = (
            np.zeros(len(indices)),
            np.zeros(len(indices)),
            [None] * len(indices),
        )

        for i, idx in enumerate(indices):
            # Perform rdkit minimization
            try:
                ff = AllChem.MMFFGetMoleculeForceField(
                    self.molecules[idx],
                    AllChem.MMFFGetMoleculeProperties(self.molecules[idx]),
                    confId=0,
                )
                ff.Initialize()
                not_converged[i] = ff.Minimize(maxIts=max_its)
            except Exception as e:
                print("Bad SMILES! Unable to minimize.")
            energies[i] = get_rdkit_energy(self.molecules[idx])
            forces[i] = get_rdkit_force(self.molecules[idx])

        return not_converged, energies, forces

    def get_rewards(self, new_energies, indices=None):
        if indices is None:
            indices = np.arange(self.n_parallel)
        assert len(new_energies) == len(indices)

        new_energies_ = np.copy(self.initial_energies)
        new_energies_[indices] = new_energies

        rewards = self.initial_energies - new_energies_
        self.initial_energies = new_energies_
        return rewards

    def initialize_molecules(self, indices, smiles_list, molecules, max_its=0):
        for i, smiles, molecule in zip(indices, smiles_list, molecules):
            # Calculate initial rdkit energy
            if smiles is not None:
                # Initialize molecule from Smiles
                self.molecules[i] = MolFromSmiles(smiles)
                self.molecules[i] = AddHs(self.molecules[i])
                # Add random conformer
                self.molecules[i].AddConformer(
                    Conformer(len(molecule.get_atomic_numbers()))
                )
            else:
                raise ValueError(
                    "Unknown molecule type {}".format(str(molecule.symbols))
                )
            self.update_coordinates_fn(self.molecules[i], molecule.get_positions())
        _, initial_energies, forces = self.calculate_energies_forces(max_its, indices)

        # Set initial energies for new molecules
        self.initial_energies[indices] = initial_energies

        # Set forces
        for i, force in zip(indices, forces):
            self.forces[i] = force


class DFTOracle(BaseOracle):
    def __init__(
        self,
        n_parallel,
        update_coordinates_fn,
        n_threads,
        converter,
        host_file_path,
    ):
        super().__init__(n_parallel, update_coordinates_fn)

        self.previous_molecules = [None] * self.n_parallel
        self.dft_server_destinations = get_dft_server_destinations(
            n_threads, host_file_path
        )
        method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        self.executors = [
            concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context(method)
            )
            for _ in range(len(self.dft_server_destinations))
        ]
        self.converter = converter
        self.tasks = {}
        self.number_processed_conformations = 0

        self.task_queue_full_flag = False

    def update_coordinates(self, positions, indices=None):
        if indices is None:
            indices = np.arange(self.n_parallel)

        # Update previous molecules
        for i in indices:
            self.previous_molecules[i] = self.molecules[i].copy()

        # Update current molecules
        super().update_coordinates(positions, indices)

    def close_executors(self):
        for executor in self.executors:
            executor.shutdown(wait=False, cancel_futures=True)

    def get_data(self, eval=False):
        assert self.task_queue_full_flag, "The task queue has not filled up yet!"

        # Wait for all computations to finish
        results = self.wait_tasks(eval=eval)
        if eval:
            assert len(results) == self.n_parallel
        else:
            assert len(results) > 0

        _, energies, forces, obs, initial_energies = zip(*results)

        energies = np.array(energies)
        forces = [np.array(force) for force in forces]
        obs = _atoms_collate_fn(obs)
        episode_total_delta_energies = np.array(initial_energies) - energies

        self.task_queue_full_flag = False
        return obs, energies, forces, episode_total_delta_energies

    def initialize_molecules(self, indices, molecules, initial_energies, forces):
        no_initial_energy_indices = []
        for i, molecule, initial_energy, force in zip(
            indices, molecules, initial_energies, forces
        ):
            self.molecules[i] = molecule.copy()
            self.previous_molecules[i] = molecule.copy()
            if initial_energy is not None:
                self.initial_energies[i] = initial_energy
                self.forces[i] = force
            else:
                no_initial_energy_indices.append(i)

        # Calculate initial DFT energy and forces if it's not provided
        if no_initial_energy_indices:
            self.submit_tasks(no_initial_energy_indices)
            # Make sure there were no unfinished tasks
            assert len(self.tasks) == len(indices)

            results = self.wait_tasks()
            assert len(results) == len(indices)

            # Update initial energies and forces
            for result in results:
                i, energy, force = result[:3]
                self.initial_energies[i] = energy
                self.forces[i] = force

    def submit_tasks(self, indices):
        self.submitted_indices = indices
        for i in indices:
            # Replace early_stop_steps with 0
            new_task = (i, 0, self.previous_molecules[i].copy())

            # Select worker and submit task
            worker_id = self.number_processed_conformations % len(
                self.dft_server_destinations
            )
            host, port = self.dft_server_destinations[worker_id]
            future = self.executors[worker_id].submit(
                calculate_dft_energy_tcp_client,
                new_task,
                host,
                port,
                False,
            )

            # Store information about conformation
            self.tasks[self.number_processed_conformations] = {
                "future": future,
                "initial_energy": self.initial_energies[i],
                "obs": self.converter(self.previous_molecules[i]),
            }
            self.number_processed_conformations += 1

            # Check if the task queue is full
            if len(self.tasks) >= self.n_parallel:
                self.task_queue_full_flag = True
                break

    def wait_tasks(self, eval=False):
        results = []

        done_task_ids = []
        # Wait for all active tasks to finish
        for key, task in self.tasks.items():
            done_task_ids.append(key)
            future = task["future"]
            obs = task["obs"]
            initial_energy = task["initial_energy"]
            i, _, energy, force = future.result()
            if energy is None:
                print(
                    f"DFT did not converged for {self.molecules[i].symbols}, id: {i}",
                    flush=True,
                )
                # If eval is True return initial_energy to correctly detect optimization failures,
                # else skip the conformation to avoid adding incorrect forces to RB.
                if eval:
                    energy = initial_energy
                else:
                    continue
            results.append((i, energy, force, obs, initial_energy))

        if not eval:
            print(f"Total conformations added: {len(results)}")

        # Delete all finished tasks
        for key in done_task_ids:
            del self.tasks[key]

        return results


class EnergyWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        dft=False,
        n_threads=1,
        minimize_on_every_step=False,
        minimize_on_done=True,
        evaluation=False,
        terminate_on_negative_reward=False,
        max_num_negative_rewards=1,
        host_file_path=None,
    ):
        # Set arguments
        self.dft = dft
        self.n_threads = n_threads
        self.minimize_on_every_step = minimize_on_every_step
        self.minimize_on_done = minimize_on_done
        self.evaluation = evaluation
        self.terminate_on_negative_reward = terminate_on_negative_reward
        self.max_num_negative_rewards = max_num_negative_rewards

        # Initialize environemnt
        super().__init__(env)
        self.n_parallel = self.env.n_parallel

        # Initialize rdkit oracle
        self.rdkit_oracle = RdkitOracle(
            n_parallel=self.n_parallel, update_coordinates_fn=set_coordinates
        )

        # Initialize DFT oracle
        converter = AtomsConverter(
            neighbor_list=ASENeighborList(cutoff=math.inf),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.dft_oracle = DFTOracle(
            n_parallel=self.n_parallel,
            update_coordinates_fn=update_ase_atoms_positions,
            n_threads=self.n_threads,
            converter=converter,
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
        self.rdkit_oracle.update_coordinates(new_positions)
        if self.dft:
            self.dft_oracle.update_coordinates(new_positions)

        # Calculate rdkit energies and forces
        calc_rdkit_energy_indices = np.where(
            self.minimize_on_every_step | (self.minimize_on_done & dones)
        )[0]

        _, new_energies, forces = self.rdkit_oracle.calculate_energies_forces(
            indices=calc_rdkit_energy_indices
        )

        # Update current energies and forces. Calculate reward
        self.rdkit_oracle.update_forces(forces, calc_rdkit_energy_indices)
        rdkit_rewards = self.rdkit_oracle.get_rewards(
            new_energies, calc_rdkit_energy_indices
        )

        # When agent encounters 'max_num_negative_rewards' terminate the episode
        self.negative_rewards_counter[rdkit_rewards < 0] += 1
        dones[self.negative_rewards_counter >= self.max_num_negative_rewards] = True

        # Log final energies of molecules
        info["final_energy"] = self.rdkit_oracle.initial_energies

        # DFT rewards
        if self.dft:
            # Conformations whose energy w.r.t. to Rdkit's MMFF is higher than
            # RDKIT_ENERGY_THRESH are highly improbable and likely to cause
            # an error in DFT calculation and/or significantly
            # slow them down. To mitigate this we propose to replace the DFT reward
            # in such states with the Rdkit reward, as they are strongly correlated in such states.
            rdkit_energy_thresh_exceeded = (
                self.rdkit_oracle.initial_energies >= RDKIT_ENERGY_THRESH
            )

            # Calculate energy and forces with DFT only for terminal states.
            # Skip conformations with energy higher than RDKIT_ENERGY_THRESH
            calculate_dft_energy_env_ids = np.where(
                self.minimize_on_done & dones & ~rdkit_energy_thresh_exceeded
            )[0]
            if len(calculate_dft_energy_env_ids) > 0:
                info["calculate_dft_energy_env_ids"] = calculate_dft_energy_env_ids
            self.dft_oracle.submit_tasks(calculate_dft_energy_env_ids)

        return obs, rdkit_rewards, dones, info

    def reset(self, indices=None):
        obs = self.env.reset(indices=indices)
        if indices is None:
            indices = np.arange(self.n_parallel)

        # Reset negative rewards counter
        self.negative_rewards_counter[indices] = 0

        # Get sizes of molecules
        smiles_list = [self.env.smiles[i] for i in indices]
        molecules = [self.env.atoms[i].copy() for i in indices]
        self.rdkit_oracle.initialize_molecules(indices, smiles_list, molecules)

        if self.dft:
            dft_initial_energies = [self.env.energy[i] for i in indices]
            dft_forces = [self.env.force[i] for i in indices]
            self.dft_oracle.initialize_molecules(
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
            obs_list.append(self.env.converter(molecule))

        self.rdkit_oracle.initialize_molecules(indices, smiles_list, molecules, max_its)

        if self.dft:
            self.dft_oracle.initialize_molecules(
                indices, molecules, energy_list, force_list
            )

        obs = _atoms_collate_fn(obs_list)
        return obs

    def update_timelimit(self, tl):
        return self.env.update_timelimit(tl)

    def get_forces(self, indices=None):
        if self.dft:
            return self.dft_oracle.get_forces(indices)
        else:
            return self.rdkit_oracle.get_forces(indices)

    def get_energies(self, indices=None):
        if self.dft:
            return self.dft_oracle.get_energies(indices)
        else:
            return self.rdkit_oracle.get_energies(indices)
