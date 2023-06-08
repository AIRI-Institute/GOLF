import math

import torch
from schnetpack import properties
from schnetpack.nn import scatter_add
from torch.nn.functional import mse_loss

from AL import DEVICE
from AL.utils import (
    calculate_gradient_norm,
    get_lr_scheduler,
    get_atoms_indices_range,
    get_optimizer_class,
)


class AL(object):
    def __init__(
        self,
        policy,
        lr,
        batch_size=256,
        clip_value=None,
        lr_scheduler=None,
        energy_loss_coef=0.01,
        force_loss_coef=0.99,
        total_steps=0,
        optimizer_name="adam",
    ):
        self.actor = policy.actor
        self.optimizer = get_optimizer_class(optimizer_name)(
            self.actor.parameters(), lr=lr
        )
        self.use_lr_scheduler = lr_scheduler is not None
        if self.use_lr_scheduler:
            lr_kwargs = {
                "gamma": 0.1,
                "total_steps": total_steps,
                "final_div_factor": 1e3,
            }
            lr_kwargs["initial_lr"] = lr
            self.lr_scheduler = get_lr_scheduler(
                lr_scheduler, self.optimizer, **lr_kwargs
            )

        self.batch_size = batch_size
        self.clip_value = clip_value
        self.energy_loss_coef = energy_loss_coef
        self.force_loss_coef = force_loss_coef
        self.total_it = 0

    def update(self, replay_buffer, *args):
        metrics = dict()
        state, force, energy = replay_buffer.sample(self.batch_size)
        output = self.actor(state, train=True)
        predicted_energy = output["energy"]
        predicted_force = output["anti_gradient"]
        n_atoms = state[properties.n_atoms]

        energy_loss = mse_loss(predicted_energy, energy.squeeze(1))
        force_loss = torch.sum(
            scatter_add(
                mse_loss(predicted_force, force, reduction="none").mean(-1),
                state[properties.idx_m],
                dim_size=n_atoms.size(0),
            )
            / n_atoms
        ) / n_atoms.size(0)
        loss = self.force_loss_coef * force_loss + self.energy_loss_coef * energy_loss

        metrics["loss"] = loss.item()
        metrics["energy_loss"] = energy_loss.item()
        metrics["force_loss"] = force_loss.item()
        metrics["energy_loss_contrib"] = energy_loss.item() * self.energy_loss_coef
        metrics["force_loss_contrib"] = force_loss.item() * self.force_loss_coef

        if not torch.all(torch.isfinite(loss)):
            print(f"Non finite values in loss")
            return metrics

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_value is not None:
            metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.clip_value
            ).item()
        else:
            metrics["grad_norm"] = calculate_gradient_norm(self.actor).item()

        if not math.isfinite(metrics["grad_norm"]):
            print("Non finite values in GD grad_norm")
            return metrics

        self.optimizer.step()

        # Update lr
        if self.use_lr_scheduler:
            self.lr_scheduler.step()

        self.total_it += 1
        return metrics

    def save(self, filename):
        self.light_save(filename)
        torch.save(self.optimizer.state_dict(), f"{filename}_optimizer")

    def light_save(self, filename):
        torch.save(self.actor.state_dict(), f"{filename}_actor")

    def load(self, filename):
        self.light_load(filename)
        self.optimizer.load_state_dict(torch.load(f"{filename}_optimizer"))

    def light_load(self, filename):
        self.actor.load_state_dict(torch.load(f"{filename}_actor"))
