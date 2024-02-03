import json
import sys
import warnings

import numpy as np
import os

from collections import deque

try:
    import wandb
except ImportError:
    pass


class Logger:
    def __init__(self, experiment_folder, config):
        # If training is restarted log to the same directory
        if os.path.exists(experiment_folder) and config.load_model is None:
            raise Exception("Experiment folder exists, apparent seed conflict!")
        if config.load_model is None:
            os.makedirs(experiment_folder)

        # If training is restarted truncate metrics file
        # to the last checkpoint
        self.metrics_file = experiment_folder / "metrics.json"
        if config.load_model:
            # Load config file from checkpoint to correctly estimate eval_freq
            with open(experiment_folder / "config.json", "r") as old_config_file:
                old_config = json.load(old_config_file)
            with open(self.metrics_file, "rb") as f:
                lines = f.readlines()
            true_eval_freq = old_config["n_parallel"] * (
                old_config["eval_freq"] // old_config["n_parallel"]
            )
            checkpoint_iter = (
                int(config.load_model.split("/")[-1].split("_")[-1]) // true_eval_freq
            )
            N = len(lines) - checkpoint_iter
            with open(self.metrics_file, "wb") as f:
                if N > 0:
                    f.writelines(lines[:-N])
                elif N == 0:
                    f.writelines(lines)
                else:
                    warnings.warn(
                        "Checkpoint iteration is older that the latest record in 'metrics.json'."
                    )
                    f.writelines(lines)
        else:
            self.metrics_file.touch()

        with open(experiment_folder / "config.json", "w") as config_file:
            json.dump(config.__dict__, config_file)

        if config.__dict__["reward"] == "dft":
            self._keep_n_episodes = config.__dict__["n_parallel"]
        else:
            self._keep_n_episodes = 10
        self.exploration_episode_lengths = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_rdkit_returns = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_dft_returns = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_final_energy = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_number = 0

        self.use_wandb = "wandb" in sys.modules  # and os.environ.get("WANDB_API_KEY")
        if self.use_wandb:
            wandb.init(project=config.project_name, save_code=True, config=config)
        else:
            warnings.warn("Could not configure wandb access.")

    def log(self, metrics):
        metrics["Exploration episodes number"] = self.exploration_episode_number
        for name, d in zip(
            [
                "episode length",
                "episode rdkit return",
                "episode dft return",
                "episode final energy",
            ],
            [
                self.exploration_episode_lengths,
                self.exploration_episode_rdkit_returns,
                self.exploration_episode_dft_returns,
                self.exploration_episode_final_energy,
            ],
        ):
            if len(d) == 0:
                mean = 0.0
                std = 0.0
            else:
                mean = np.mean(d)
                std = np.std(d)
            metrics[f"Exploration {name}, mean"] = mean
            metrics[f"Exploration {name}, std"] = std
        with open(self.metrics_file, "a") as out_metrics:
            json.dump(metrics, out_metrics)
            out_metrics.write("\n")

        if self.use_wandb:
            wandb.log(metrics)

    def update_evaluation_statistics(
        self,
        episode_length,
        episode_rdkit_return,
        episode_final_energy,
    ):
        self.exploration_episode_number += 1
        self.exploration_episode_lengths.append(episode_length)
        self.exploration_episode_rdkit_returns.append(episode_rdkit_return)
        self.exploration_episode_final_energy.append(episode_final_energy)

    def update_dft_return_statistics(self, episode_dft_return):
        for val in episode_dft_return:
            self.exploration_episode_dft_returns.append(val)
