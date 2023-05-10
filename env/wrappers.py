import os
from collections import defaultdict

import gym
import numpy as np
from rdkit.Chem import AddHs, AllChem, Conformer, MolFromSmiles
from schnetpack.data.loader import _atoms_collate_fn

from .dft import (
    calculate_dft_energy_queue,
    get_dft_forces_energy,
    update_ase_atoms_positions,
)
from .moldynamics_env import MolecularDynamics
from .xyz2mol import get_rdkit_energy, get_rdkit_force, parse_molecule, set_coordinates

RDKIT_ENERGY_THRESH = 500


class RewardWrapper(gym.Wrapper):
    molecules_xyz = {
        "C7O3C2OH8": "aspirin.xyz",
        "N2C12H10": "azobenzene.xyz",
        "C6H6": "benzene.xyz",
        "C2OH6": "ethanol.xyz",
        "C3O2H4": "malonaldehyde.xyz",
        "C10H8": "naphthalene.xyz",
        "C2ONC4OC2H9": "paracetamol.xyz",
        "C3OC4O2H6": "salicylic_acid.xyz",
        "C7H8": "toluene.xyz",
        "C2NCNCO2H4": "uracil.xyz",
    }

    def __init__(
        self,
        env,
        dft=False,
        n_threads=1,
        minimize_on_every_step=False,
        molecules_xyz_prefix="",
        terminate_on_negative_reward=False,
        max_num_negative_rewards=1,
    ):
        # Set arguments
        self.dft = dft
        self.n_threads = n_threads
        self.minimize_on_every_step = minimize_on_every_step
        self.molecules_xyz_prefix = molecules_xyz_prefix
        self.terminate_on_negative_reward = terminate_on_negative_reward
        self.max_num_negative_rewards = max_num_negative_rewards

        self.update_coordinates = {
            "rdkit": set_coordinates,
            "dft": update_ase_atoms_positions,
        }
        self.get_energy = {"rdkit": get_rdkit_energy, "dft": get_dft_forces_energy}
        self.get_force = {
            "rdkit": get_rdkit_force,
            "dft": None,
        }

        # Check parent class to name the reward correctly
        if isinstance(env, MolecularDynamics):
            self.reward_name = "env_reward"
        else:
            self.reward_name = "unknown_reward"
        super().__init__(env)

        # Initialize dictionaries
        self.n_parallel = self.env.n_parallel
        self.initial_energy = {
            "rdkit": [None] * self.n_parallel,
            "dft": [None] * self.n_parallel,
        }
        self.force = {
            "rdkit": [None] * self.n_parallel,
            "dft": [None] * self.n_parallel,
        }
        self.molecule = {
            "rdkit": [None] * self.n_parallel,
            "dft": [None] * self.n_parallel,
        }
        self.threshold_exceeded = [0.0 for _ in range(self.n_parallel)]
        self.negative_rewards_counter = [0 for _ in range(self.n_parallel)]
        self.molecules = {}
        self.parse_molecules()

    def parse_molecules(self):
        # Parse rdkit molecules
        self.molecules["rdkit"] = {}
        for formula, path in RewardWrapper.molecules_xyz.items():
            molecule = parse_molecule(os.path.join(self.molecules_xyz_prefix, path))
            # Check if the provided molecule is valid
            try:
                self.get_energy["rdkit"](molecule)
            except AttributeError:
                raise ValueError("Provided molucule was not parsed correctly")
            self.molecules["rdkit"][formula] = molecule

    def step(self, actions):
        obs, env_rewards, dones, info = super().step(actions)

        # Put rewards from the environment into info
        info = dict(info, **{self.reward_name: env_rewards})

        # Get sizes of molecules
        atoms_num = self.get_atoms_num()
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
            self.update_coordinates["rdkit"](
                self.molecule["rdkit"][idx], self.env.atoms[idx].get_positions()
            )
            if self.dft:
                self.update_coordinates["dft"](
                    self.molecule["dft"][idx], self.env.atoms[idx].get_positions()
                )

            # Calculate current rdkit reward for every trajectory
            if self.minimize_on_every_step or dones[idx]:
                (
                    not_converged[idx],
                    final_energy[idx],
                    self.force["rdkit"][idx],
                ) = self.minimize_rdkit(idx)
                rdkit_rewards[idx] = (
                    self.initial_energy["rdkit"][idx] - final_energy[idx]
                )

        # DFT rewards
        if self.dft:
            queue = []
            for idx in range(self.n_parallel):
                if self.minimize_on_every_step or dones[idx]:
                    # Rdkit reward lower than RDKIT_DELTA_THRESH indicates highly improbable
                    # conformations which are likely to cause an error in DFT calculation and/or
                    # significantly slow them down. To mitigate this we propose to replace DFT reward
                    # in such states with rdkit reward. Note that rdkit reward is strongly
                    # correlated with DFT reward and should not intefere with the training.
                    if final_energy[idx] < RDKIT_ENERGY_THRESH:
                        queue.append((self.molecule["dft"][idx], atoms_num[idx], idx))
                    else:
                        self.threshold_exceeded[idx] += 1
                        rewards[idx] = rdkit_rewards[idx]
                        self.force["dft"][idx] = self.force["rdkit"][idx]

            # Sort queue according to the molecule size
            queue = sorted(queue, key=lambda x: x[1], reverse=True)
            # TODO think about M=None, etc.
            result = calculate_dft_energy_queue(queue, n_threads=self.n_threads)
            for idx, _, energy, force in result:
                rewards[idx] = self.initial_energy["dft"][idx] - energy
                self.force["dft"][idx] = force
        else:
            rewards = rdkit_rewards

        # Dones and info
        for idx in range(self.n_parallel):
            # If minimize_on_every step update initial energy
            if self.minimize_on_every_step:
                # initial_energy = final_energy
                self.initial_energy["rdkit"][idx] -= rdkit_rewards[idx]
                # FIXME ?
                # At the moment the wrapper is guaranteed to work correctly
                # only with done_when_not_improved=True. In case of greedy=True
                # we might get final_energy > RDKIT_ENERGY_THRESH on steps
                # [t, ..., t + T - 1] and then get final_energy > RDKIT_ENERGY_THRESH on
                # step t + T (although this is highly unlikely, it is possible).
                # Then the initial DFT energy would be calculated from the
                # rdkit reward but the final energy would come from DFT.
                if self.dft:
                    self.initial_energy["dft"][idx] = (
                        self.initial_energy["dft"][idx] - rewards[idx]
                    )

            # When agent encountered 'max_num_negative_rewards'
            # terminate the episode
            if self.terminate_on_negative_reward:
                if rewards[idx] <= 0:
                    self.negative_rewards_counter[idx] += 1
                if self.negative_rewards_counter[idx] >= self.max_num_negative_rewards:
                    dones[idx] = True

            # Log final energy of the molecule
            stats_done["final_energy"][idx] = final_energy[idx]
            stats_done["not_converged"][idx] = not_converged[idx]

            # Log percentage of times in which treshold was  exceeded
            if self.dft:
                threshold_exceeded_pct[idx] = (
                    self.threshold_exceeded[idx] / env_steps[idx]
                )
            else:
                threshold_exceeded_pct[idx] = 0
            stats_done["threshold_exceeded_pct"][idx] = threshold_exceeded_pct[idx]
            stats_done["final_rl_energy"][idx] = final_energy[idx]

        # Compute mean of stats over finished trajectories and update info
        info = dict(info, **stats_done)

        return obs, rewards, np.stack(dones), info

    def reset(self, indices=None):
        obs = self.env.reset(indices=indices)
        if indices is None:
            indices = np.arange(self.n_parallel)

        # Get sizes of molecules
        atoms_num = self.get_atoms_num()

        for idx in indices:
            # Reset negative rewards counter
            self.negative_rewards_counter[idx] = 0

            # Calculate initial rdkit energy
            if self.env.smiles[idx] is not None:
                # Initialize molecule from Smiles
                self.molecule["rdkit"][idx] = MolFromSmiles(self.env.smiles[idx])
                self.molecule["rdkit"][idx] = AddHs(self.molecule["rdkit"][idx])
                # Add random conformer
                self.molecule["rdkit"][idx].AddConformer(Conformer(atoms_num[idx]))
            elif str(self.env.atoms[idx].symbols) in self.molecules["rdkit"]:
                self.molecule["rdkit"][idx] = self.molecules["rdkit"][
                    str(self.env.atoms[idx].symbols)
                ]
            else:
                raise ValueError(
                    "Unknown molecule type {}".format(str(self.env.atoms[idx].symbols))
                )
            self.update_coordinates["rdkit"](
                self.molecule["rdkit"][idx], self.env.atoms[idx].get_positions()
            )
            (
                _,
                self.initial_energy["rdkit"][idx],
                self.force["rdkit"][idx],
            ) = self.minimize_rdkit(idx)
            # Calculate initial dft energy
            if self.dft:
                queue = []
                self.threshold_exceeded[idx] = 0
                # psi4.core.Molecule object cannot be stored in the MP Queue.
                # Instead store ase.Atoms and transform into psi4 format later.
                self.molecule["dft"][idx] = self.env.atoms[idx].copy()
                if self.env.energy[idx] is not None:
                    self.initial_energy["dft"][idx] = self.env.energy[idx]
                    self.force["dft"][idx] = self.env.force[idx]
                else:
                    queue.append((self.molecule["dft"][idx], atoms_num[idx], idx))

        # Calculate initial dft energy if it is not provided
        if self.dft and len(queue) > 0:
            # Sort queue according to the molecule size
            queue = sorted(queue, key=lambda x: x[1], reverse=True)
            # TODO think about M=None, etc.
            result = calculate_dft_energy_queue(queue, n_threads=self.n_threads)
            for idx, _, energy, force in result:
                self.initial_energy["dft"][idx] = energy
                self.force["dft"][idx] = force

        return obs

    def set_initial_positions(
        self, atoms_list, smiles_list, energy_list, force_list, max_its=0
    ):
        super().reset(increment_conf_idx=False)

        # Set molecules and get observation
        obs_list = []
        for idx, (atoms, smiles, energy, force) in enumerate(
            zip(atoms_list, smiles_list, energy_list, force_list)
        ):
            self.env.atoms[idx] = atoms.copy()
            obs_list.append(self.env.converter(self.env.atoms[idx]))

            # Reset negative rewards counter
            self.negative_rewards_counter[idx] = 0

            # Calculate initial rdkit energy
            if smiles is not None:
                # Initialize molecule from Smiles
                self.molecule["rdkit"][idx] = MolFromSmiles(smiles)
                self.molecule["rdkit"][idx] = AddHs(self.molecule["rdkit"][idx])
                # Add random conformer
                self.molecule["rdkit"][idx].AddConformer(
                    Conformer(len(atoms.get_atomic_numbers()))
                )
            elif str(self.env.atoms[idx].symbols) in self.molecules["rdkit"]:
                self.molecule["rdkit"][idx] = self.molecules["rdkit"][
                    str(self.env.atoms[idx].symbols)
                ]
            else:
                raise ValueError(
                    "Unknown molecule type {}".format(str(self.env.atoms[idx].symbols))
                )
            self.update_coordinates["rdkit"](
                self.molecule["rdkit"][idx], self.env.atoms[idx].get_positions()
            )
            (
                _,
                self.initial_energy["rdkit"][idx],
                self.force["rdkit"][idx],
            ) = self.minimize_rdkit(idx, max_its)

            # Calculate initial dft energy
            if self.dft:
                queue = []
                self.threshold_exceeded[idx] = 0
                # psi4.core.Molecule object cannot be stored in the MP Queue.
                # Instead store ase.Atoms and transform into psi4 format later.
                self.molecule["dft"][idx] = self.env.atoms[idx].copy()
                if energy is not None and force is not None:
                    self.initial_energy["dft"][idx] = energy
                    self.force["dft"][idx] = force
                else:
                    queue.append(
                        (
                            self.molecule["dft"][idx],
                            len(atoms.get_atomic_numbers()),
                            idx,
                        )
                    )

        # Calculate initial dft energy if it is not provided
        if self.dft and len(queue) > 0:
            # Sort queue according to the molecule size
            queue = sorted(queue, key=lambda x: x[1], reverse=True)
            # TODO think about M=None, etc.
            result = calculate_dft_energy_queue(queue, n_threads=self.n_threads)
            for idx, _, energy, force in result:
                self.initial_energy["dft"][idx] = energy
                self.force["dft"][idx] = force

        obs = _atoms_collate_fn(obs_list)
        return obs

    def minimize_rdkit(self, idx, max_its=0):
        # Perform rdkit minimization
        ff = AllChem.MMFFGetMoleculeForceField(
            self.molecule["rdkit"][idx],
            AllChem.MMFFGetMoleculeProperties(self.molecule["rdkit"][idx]),
            confId=0,
        )
        ff.Initialize()
        not_converged = ff.Minimize(maxIts=max_its)
        energy = self.get_energy["rdkit"](self.molecule["rdkit"][idx])
        force = self.get_force["rdkit"](self.molecule["rdkit"][idx])

        return not_converged, energy, force

    def get_atoms_num(self):
        cumsum_atoms_num = np.asarray(self.get_atoms_num_cumsum())
        return (cumsum_atoms_num[1:] - cumsum_atoms_num[:-1]).tolist()

    def get_env_step(self):
        return self.env.get_env_step()

    def update_timelimit(self, tl):
        return self.env.update_timelimit(tl)

    def get_forces(self, indices=None):
        if indices is None:
            indices = np.arange(self.n_parallel)
        if self.dft:
            reward_name = "dft"
        else:
            reward_name = "rdkit"
        return [
            np.array(self.force[reward_name][ind], dtype=np.float32) for ind in indices
        ]

    def get_energies(self, indices=None):
        if indices is None:
            indices = np.arange(self.n_parallel)
        if self.dft:
            reward_name = "dft"
        else:
            reward_name = "rdkit"
        return np.array(
            [self.initial_energy[reward_name][idx] for idx in indices], dtype=np.float32
        )
