import math
import warnings
from sqlite3 import DatabaseError

import backoff
import gymnasium as gym
import numpy as np
import torch
from ase.db import connect
from schnetpack.interfaces import AtomsConverter
from schnetpack.transform import ASENeighborList

np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# For backoff exceptions
def on_giveup(details):
    print(
        "Giving Up after {} tries. Time elapsed: {:.3f} :(".format(
            details["tries"], details["elapsed"]
        )
    )


class MolecularDynamics(gym.Env):
    metadata = {"render_modes": ["human"], "name": "md_v0"}
    DISTANCE_THRESH = 0.7

    def __init__(
        self,
        db_path,
        converter,
        n_parallel=1,
        timelimit=10,
        sample_initial_conformations=True,
        num_initial_conformations=50000,
    ):
        self.db_path = db_path
        self.converter = converter
        self.n_parallel = n_parallel
        self.TL = timelimit
        self.sample_initial_conformations = sample_initial_conformations

        self.db_len = self.get_db_length()
        self.atoms = None
        self.mean_energy = 0.0
        self.std_energy = 1.0
        self.initial_molecule_conformations = []
        self.initial_conformations_ids = []

        # Store random subset of molecules DB
        self.get_initial_molecule_conformations(num_initial_conformations)
        self.conformation_idx = 0

        # Initialize lists
        self.atoms = [None] * self.n_parallel
        self.smiles = [None] * self.n_parallel
        self.energy = [None] * self.n_parallel
        self.optimal_energy = [None] * self.n_parallel
        self.force = [None] * self.n_parallel
        self.env_steps = [None] * self.n_parallel
        self.atoms_ids = [None] * self.n_parallel

        self.total_num_bad_pairs_before = 0
        self.total_num_bad_pairs_after = 0

    def step(self, actions):
        # Get number of atoms in each molecule
        cumsum_numbers_atoms = self.get_atoms_num_cumsum()

        obs = []
        rewards = [None] * self.n_parallel
        dones = [None] * self.n_parallel
        info = {}

        for idx in range(self.n_parallel):
            # Unpad action
            self.atoms[idx].set_positions(
                self.atoms[idx].get_positions()
                + actions[cumsum_numbers_atoms[idx] : cumsum_numbers_atoms[idx + 1]]
            )

            # Check if there are atoms too close to each other in the molecule
            (
                self.atoms[idx],
                num_bad_pairs_before,
                num_bad_pairs_after,
            ) = self.process_molecule(self.atoms[idx])
            self.total_num_bad_pairs_before += num_bad_pairs_before
            self.total_num_bad_pairs_after += num_bad_pairs_after

            # Terminate the episode if TL is reached
            self.env_steps[idx] += 1
            dones[idx] = self.env_steps[idx] >= self.TL

        # Add info about bad pairs
        info["total_bad_pairs_before_processing"] = self.total_num_bad_pairs_before
        info["total_bad_pairs_after_processing"] = self.total_num_bad_pairs_after

        # Collate observations into a batch
        obs = self.converter(self.atoms)

        return obs, rewards, dones, info

    def reset(self, indices=None, increment_conf_idx=True):
        # If indices is not provided reset all molecules
        if indices is None:
            indices = np.arange(self.n_parallel)

        # If sample_initial_conformations iterate over all initial conformations sequentially
        if self.sample_initial_conformations:
            db_indices = np.random.choice(
                len(self.initial_molecule_conformations), len(indices), replace=False
            )
        else:
            start_conf_idx = self.conformation_idx % len(
                self.initial_molecule_conformations
            )
            db_indices = np.mod(
                np.arange(start_conf_idx, start_conf_idx + len(indices)),
                len(self.initial_conformations_ids),
            ).astype(np.int64)
            if increment_conf_idx:
                self.conformation_idx += len(indices)

        rows = [self.initial_molecule_conformations[db_idx] for db_idx in db_indices]

        for idx, row, atom_id in zip(
            indices, rows, self.initial_conformations_ids[db_indices]
        ):
            # Copy to avoid changing the atoms object inplace
            self.atoms[idx] = row.toatoms().copy()
            self.atoms_ids[idx] = int(atom_id)

            # Check if row has Smiles
            if hasattr(row, "smiles"):
                self.smiles[idx] = row.smiles

            # Energy and optimal_energy in Hartrees.
            # Energies and optimal energies can sometimes be stored in different formats.
            if hasattr(row.data, "energy"):
                if isinstance(row.data["energy"], list):
                    assert len(row.data["energy"]) == 1
                    self.energy[idx] = row.data["energy"][0]
                elif isinstance(row.data["energy"], np.ndarray):
                    assert len(row.data["energy"]) == 1
                    self.energy[idx] = row.data["energy"].item()
                else:
                    self.energy[idx] = row.data["energy"]
            # In case the database is the result of optimization
            elif hasattr(row.data, "initial_energy"):
                self.energy[idx] = row.data["initial_energy"]

            if hasattr(row.data, "optimal_energy"):
                if isinstance(row.data["optimal_energy"], list):
                    assert len(row.data["optimal_energy"]) == 1
                    self.optimal_energy[idx] = row.data["optimal_energy"][0]
                elif isinstance(row.data["optimal_energy"], np.ndarray):
                    assert len(row.data["optimal_energy"]) == 1
                    self.optimal_energy[idx] = row.data["optimal_energy"].item()
                else:
                    self.optimal_energy[idx] = row.data["optimal_energy"]

            # forces in Hartees/ Angstrom
            if hasattr(row.data, "forces"):
                self.force[idx] = row.data["forces"]
            elif hasattr(row.data, "final_forces"):
                # In case the database is the result of optimization
                self.force[idx] = row.data["final_forces"]

            # Reset env_steps
            self.env_steps[idx] = 0

        # Collate observations into a batch
        obs = self.converter([self.atoms[idx] for idx in indices])

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
            self.initial_conformations_ids = np.arange(1, self.db_len + 1)
        else:
            self.initial_conformations_ids = np.random.choice(
                np.arange(1, self.db_len + 1),
                min(self.db_len, num_initial_conformations),
                replace=False,
            )
        self.initial_molecule_conformations = []
        for idx in self.initial_conformations_ids:
            row = self.get_molecule(int(idx))
            self.initial_molecule_conformations.append(row)

    # Makes sqllite3 database compatible with NFS storages
    @backoff.on_exception(
        backoff.expo,
        exception=DatabaseError,
        jitter=backoff.full_jitter,
        max_tries=10,
        on_giveup=on_giveup,
    )
    def get_molecule(self, idx):
        with connect(self.db_path) as conn:
            return conn.get(idx)

    def get_atoms_num_cumsum(self):
        atoms_num_cumsum = [0]
        for atom in self.atoms:
            atoms_num_cumsum.append(
                atoms_num_cumsum[-1] + len(atom.get_atomic_numbers())
            )

        return atoms_num_cumsum

    def get_bad_pairs_indices(self, positions):
        dir_ij = positions[None, :, :] - positions[:, None, :]
        r_ij = np.linalg.norm(dir_ij, axis=2)

        # Set diagonal elements to a large positive number
        r_ij[np.diag_indices_from(r_ij)] = 10.0
        dir_ij /= r_ij[..., None]

        # Set lower triangle matrix of r_ij to a large positive number
        # to avoid finding dublicate pairs
        r_ij[np.tri(r_ij.shape[0], k=-1).astype(bool)] = 10.0

        return np.argwhere(r_ij < MolecularDynamics.DISTANCE_THRESH), dir_ij, r_ij

    def process_molecule(self, atoms):
        new_atoms = atoms.copy()
        positions = new_atoms.get_positions()

        # Detect atoms too close to each other
        bad_indices_before, dir_ij, r_ij = self.get_bad_pairs_indices(positions)

        # Move atoms apart. At the moment we assume
        # that r_ij < THRESH is a rare event that affects at most
        # one pair of atoms so moving them apart should not cause
        # any other r_kl to become < THRESH.
        for i, j in bad_indices_before:
            coef = MolecularDynamics.DISTANCE_THRESH + 0.05 - r_ij[i, j]
            positions[i] -= 0.5 * coef * dir_ij[i, j]
            positions[j] -= 0.5 * coef * dir_ij[j, i]
        new_atoms.set_positions(positions)

        # Check if assumption does not hold
        bad_indices_after, _, _ = self.get_bad_pairs_indices(positions)

        return new_atoms, len(bad_indices_before), len(bad_indices_after)

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        return seed


def env_fn(**kwargs):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    converter = AtomsConverter(
        neighbor_list=ASENeighborList(cutoff=math.inf),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    env = MolecularDynamics(converter=converter, **kwargs)
    return env
