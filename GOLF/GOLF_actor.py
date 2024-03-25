import numpy as np
import torch
import torch.nn as nn

from GOLF.models import NeuralNetworkPotentials
from GOLF.utils import get_atoms_indices_range, get_n_atoms

KCALMOL_2_HARTREE = 627.5

EPS = 1e-8


class Actor(nn.Module):
    def __init__(
        self,
        nnp_type,
        nnp_args,
        forces_norm_limit=None,
    ):
        super(Actor, self).__init__()
        self.forces_norm_limit = forces_norm_limit

        self.nnp = NeuralNetworkPotentials[nnp_type](**nnp_args)

        self.last_energy = None
        self.last_forces = None

    def _limit_forces_norm(self, forces, n_atoms):
        if self.forces_norm_limit is None:
            return forces

        coefficient = torch.ones(
            size=(forces.size(0), 1), dtype=torch.float32, device=forces.device
        )
        for molecule_id in range(n_atoms.size(0) - 1):
            max_norm = (
                torch.linalg.vector_norm(
                    forces[n_atoms[molecule_id] : n_atoms[molecule_id + 1]],
                    dim=-1,
                    keepdims=True,
                )
                .max(dim=1, keepdims=True)
                .values
            )
            max_norm = torch.maximum(
                max_norm, torch.full_like(max_norm, fill_value=EPS, dtype=torch.float32)
            )
            coefficient[n_atoms[molecule_id] : n_atoms[molecule_id + 1]] = (
                torch.minimum(
                    self.forces_norm_limit / max_norm,
                    torch.ones_like(max_norm, dtype=torch.float32),
                )
            )

        return forces * coefficient

    def _save_last_output(self, energy, forces):
        self.last_energy = energy.detach().clone()
        self.last_forces = forces.detach().clone()

    def _get_last_output(self):
        if self.last_energy is None or self.last_forces is None:
            raise ValueError("Last output has not been set yet!")
        return self.last_energy, self.last_forces

    def forward(self, batch, active_optimizers_ids=None, train=False):
        energy, forces = self.nnp(batch)
        self._save_last_output(energy, forces)
        if train:
            return {"forces": forces, "energy": energy}
        forces = forces.detach()
        forces = self._limit_forces_norm(forces, get_atoms_indices_range(batch))

        return {"forces": forces, "energy": energy}


class RdkitActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def forward(self, batch, active_optimizers_ids=None, train=False):
        if active_optimizers_ids is None:
            opt_ids = list(range(self.env.n_parallel))
        else:
            opt_ids = active_optimizers_ids

        # Update atoms inside env
        current_coordinates = [
            self.env.unwrapped.atoms[idx].get_positions() for idx in opt_ids
        ]
        n_atoms = get_n_atoms(batch)
        new_coordinates = torch.split(
            batch.pos.detach().cpu(),
            n_atoms.tolist(),
        )
        assert len(new_coordinates) == len(opt_ids)
        new_coordinates = [
            np.float64(new_coordinates[i].numpy()) for i in range(len(opt_ids))
        ]
        # Update coordinates inside env
        self.env.surrogate_oracle.update_coordinates(new_coordinates, indices=opt_ids)
        energies, forces = self.env.surrogate_oracle.calculate_energies_forces(
            indices=opt_ids
        )

        # Restore original coordinates
        self.env.surrogate_oracle.update_coordinates(
            current_coordinates, indices=opt_ids
        )

        # Forces in (kcal/mol)/angstrom. Transform into hartree/angstrom.
        forces = torch.cat(
            [torch.tensor(force / KCALMOL_2_HARTREE) for force in forces]
        )

        return {"forces": forces, "energy": torch.tensor(energies)}
