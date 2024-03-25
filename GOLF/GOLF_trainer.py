import torch
from torch.nn.functional import l1_loss, mse_loss
from torch_geometric.utils import scatter
from torchmetrics.functional import mean_absolute_error

from GOLF import DEVICE
from GOLF.optim import Lion
from GOLF.utils import (
    calculate_gradient_norm,
    get_lr_scheduler,
)
from utils.utils import ignore_extra_args


optimizers = {
    "adam": ignore_extra_args(torch.optim.Adam),
    "AdamW": ignore_extra_args(torch.optim.AdamW),
    "lion": ignore_extra_args(Lion),
}


class GOLF(object):
    def __init__(
        self,
        policy,
        lr,
        batch_size=256,
        clip_value=None,
        lr_scheduler=None,
        energy_loss_coef=0.01,
        force_loss_coef=0.99,
        load_model=None,
        total_steps=0,
        utd_ratio=1,
        optimizer_name="adam",
    ):
        self.actor = policy.actor
        self.optimizer = optimizers[optimizer_name](
            params=self.actor.parameters(), lr=lr, amsgrad=True
        )
        if load_model:
            self.load(load_model)
            last_epoch = int(load_model.split("/")[-1].split("_")[-1]) * utd_ratio
        else:
            last_epoch = -1

        self.use_lr_scheduler = lr_scheduler is not None
        if self.use_lr_scheduler:
            lr_kwargs = {
                "gamma": 0.1,
                "total_steps": total_steps,
                "final_div_factor": 1e3,
                "last_epoch": last_epoch,
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

        # Train model
        batch, force, energy = replay_buffer.sample(self.batch_size)
        output = self.actor(batch, train=True)
        predicted_energy = output["energy"]
        predicted_force = output["forces"]

        energy_loss = mse_loss(predicted_energy, energy.squeeze(1))
        # energy_loss = l1_loss(predicted_energy, energy.squeeze(1))

        force_loss = torch.mean(
            scatter(
                mse_loss(predicted_force, force, reduction="none").mean(-1),
                batch.batch,
                dim_size=batch.batch_size,
                reduce="mean",
            )
        )
        loss = self.force_loss_coef * force_loss + self.energy_loss_coef * energy_loss

        if not torch.all(torch.isfinite(loss)):
            print(f"Non finite values in loss")
            return metrics

        self.optimizer.zero_grad()
        loss.backward()

        if self.clip_value is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.clip_value
            )
        else:
            grad_norm = calculate_gradient_norm(self.actor)
        if not torch.all(torch.isfinite(grad_norm)):
            print("Non finite values in GD grad_norm")
            return metrics

        self.optimizer.step()

        # Update lr
        if self.use_lr_scheduler:
            self.lr_scheduler.step()

        self.total_it += 1
        metrics["train/loss"] = loss.item()
        metrics["train/energy_loss"] = energy_loss.item()
        metrics["train/force_loss"] = force_loss.item()
        metrics["train/energy_loss_contrib"] = (
            energy_loss.item() * self.energy_loss_coef
        )
        metrics["train/force_loss_contrib"] = force_loss.item() * self.force_loss_coef
        metrics["grad_norm"] = grad_norm.item()
        if self.use_lr_scheduler:
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
        return metrics

    def eval(self, replay_buffer):
        metrics = dict()

        # Evaluate on test dataset
        batch, eval_forces, eval_energy = replay_buffer.sample_eval(self.batch_size)
        output = self.actor(batch, train=True)
        predicted_energy = output["energy"]
        predicted_force = output["forces"]

        with torch.no_grad():
            eval_energy_loss = mse_loss(predicted_energy, eval_energy.squeeze(1))
            eval_force_loss = torch.mean(
                scatter(
                    mse_loss(predicted_force, eval_forces, reduction="none").mean(-1),
                    batch.batch,
                    dim_size=batch.batch_size,
                    reduce="mean",
                )
            )
            eval_loss = (
                self.force_loss_coef * eval_force_loss
                + self.energy_loss_coef * eval_energy_loss
            )

        metrics["eval/loss"] = eval_loss.item()
        metrics["eval/energy_loss"] = eval_energy_loss.item()
        metrics["eval/energy_loss_MAE"] = mean_absolute_error(
            predicted_energy, eval_energy.squeeze(1)
        ).item()
        metrics["eval/force_loss"] = eval_force_loss.item()
        metrics["eval/force_loss_MAE"] = mean_absolute_error(
            predicted_force, eval_forces
        ).item()
        metrics["eval/energy_loss_contrib"] = (
            eval_energy_loss.item() * self.energy_loss_coef
        )
        metrics["eval/force_loss_contrib"] = (
            eval_force_loss.item() * self.force_loss_coef
        )
        return metrics

    def save(self, filename):
        self.light_save(filename)
        torch.save(self.optimizer.state_dict(), f"{filename}_optimizer")

    def light_save(self, filename):
        torch.save(self.actor.state_dict(), f"{filename}_actor")

    def load(self, filename):
        self.light_load(filename)
        self.optimizer.load_state_dict(
            torch.load(f"{filename}_optimizer", map_location=DEVICE)
        )
        self.optimizer.param_groups[0]["capturable"] = True

    def light_load(self, filename):
        self.actor.load_state_dict(torch.load(f"{filename}_actor", map_location=DEVICE))
