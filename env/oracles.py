import concurrent.futures
import copy
import multiprocessing as mp

import numpy as np
import torch
from rdkit.Chem import AddHs, AllChem, Conformer, MolFromSmiles
from torch_geometric.data import Batch, Data

from env.dft import calculate_dft_energy_tcp_client, get_dft_server_destinations
from env.xyz2mol import get_rdkit_energy, get_rdkit_force
from GOLF import DEVICE


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

        energies, forces = np.zeros(len(indices)), [None] * len(indices)

        for i, idx in enumerate(indices):
            # Perform rdkit minimization
            ff = AllChem.MMFFGetMoleculeForceField(
                self.molecules[idx],
                AllChem.MMFFGetMoleculeProperties(self.molecules[idx]),
                confId=0,
            )
            ff.Initialize()
            ff.Minimize(maxIts=max_its)
            energies[i] = get_rdkit_energy(self.molecules[idx])
            forces[i] = get_rdkit_force(self.molecules[idx])

        return energies, forces

    def get_energy_delta(self, new_energies, indices=None):
        if indices is None:
            indices = np.arange(self.n_parallel)
        assert len(new_energies) == len(indices)

        new_energies_ = np.copy(self.initial_energies)
        new_energies_[indices] = new_energies

        energy_delta = self.initial_energies - new_energies_
        self.initial_energies = new_energies_
        return energy_delta

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
        initial_energies, forces = self.calculate_energies_forces(max_its, indices)

        # Set initial energies for new molecules
        self.initial_energies[indices] = initial_energies

        # Set forces
        for i, force in zip(indices, forces):
            self.forces[i] = force


class NeuralOracle(BaseOracle):
    def __init__(self, n_parallel, update_coordinates_fn, model, tau):
        super().__init__(n_parallel, update_coordinates_fn)
        self.model = model
        self.tau = tau
        # Create empty batch
        self.batch = Batch.from_data_list(
            [
                Data(
                    z=torch.empty(size=(1, 1)).long(),
                    pos=torch.empty(size=(1, 3)).float(),
                )
                for _ in range(self.n_parallel)
            ]
        ).to(DEVICE)

    def calculate_energies_forces(self, max_its=0, indices=None):
        if indices is None:
            indices = np.arange(self.n_parallel)
        if len(indices) == 0:
            return np.zeros(len(indices)), torch.zeros(self.batch.pos.shape)
        batch = Batch.from_data_list(self.batch.index_select(indices))
        torch.set_grad_enabled(True)
        output = self.model(batch, train=True)
        return (
            output["energy"].detach().cpu().numpy(),
            output["forces"].detach().cpu(),
        )

    def get_energy_delta(self, new_energies, indices=None):
        if indices is None:
            indices = np.arange(self.n_parallel)
        assert len(new_energies) == len(indices)

        new_energies_ = np.copy(self.initial_energies)
        new_energies_[indices] = new_energies

        energy_delta = self.initial_energies - new_energies_
        self.initial_energies = new_energies_
        return energy_delta

    def initialize_molecules(self, indices, smiles_list, molecules, max_its=0):
        # Data list
        data_list = self.batch.to_data_list()
        for i, molecule in zip(indices, molecules):
            data_list[i] = Data(
                z=torch.from_numpy(molecule.get_atomic_numbers()).long(),
                pos=torch.from_numpy(molecule.get_positions()).float(),
            ).to(DEVICE)
        self.batch = Batch.from_data_list(data_list)
        initial_energies, forces = self.calculate_energies_forces(max_its, indices)

        # Set initial energies for new molecules
        self.initial_energies[indices] = initial_energies

        # Set forces
        for i, force in zip(indices, forces):
            self.forces[i] = force

    def update_coordinates(self, positions, indices=None):
        if indices is None:
            indices = np.arange(self.n_parallel)
        assert len(positions) == len(
            indices
        ), f"Not enough values to update all molecules! Expected {self.n_parallel} but got {len(positions)}"
        data_list = self.batch.to_data_list()

        for i, position in zip(indices, positions):
            data_list[i].update({"pos": torch.from_numpy(position).float().to(DEVICE)})

        self.batch = Batch.from_data_list(data_list)

    def update_forces(self, forces, indices=None):
        if indices is None:
            indices = np.arange(self.n_parallel)
        if len(indices) == 0:
            return
        n_atoms = [mol.pos.shape[0] for mol in self.batch.index_select(indices)]
        forces_list = torch.split(forces, n_atoms)
        assert len(forces_list) == len(indices)
        for i, forces in zip(indices, forces_list):
            self.forces[i] = forces.numpy()

    def update_model(self, new_model):
        for new_param, param in zip(new_model.parameters(), self.model.parameters()):
            param.data.copy_(self.tau * param.data + (1 - self.tau) * new_param.data)


class DFTOracle(BaseOracle):
    def __init__(
        self,
        n_parallel,
        update_coordinates_fn,
        n_threads,
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
        obs = Batch.from_data_list(obs)
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
                "obs": Data(
                    z=torch.from_numpy(
                        self.previous_molecules[i].get_atomic_numbers()
                    ).long(),
                    pos=torch.from_numpy(
                        self.previous_molecules[i].get_positions()
                    ).float(),
                ).to(DEVICE),
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
