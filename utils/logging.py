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
        if config.load_model is not None:
            with open(self.metrics_file, "rb") as f:
                lines = f.readlines()
            true_eval_freq = config.n_parallel * (config.eval_freq // config.n_parallel)
            checkpoint_iter = (
                int(config.load_model.split("/")[-1].split("_")[-1]) // true_eval_freq
            )
            N = len(lines) - checkpoint_iter
            with open(self.metrics_file, "wb") as f:
                f.writelines(lines[:-N])
        self.metrics_file.touch()

        with open(experiment_folder / "config.json", "w") as config_file:
            json.dump(config.__dict__, config_file)

        self._keep_n_episodes = 10
        self.exploration_episode_lengths = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_returns = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_mean_Q = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_final_energy = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_final_rl_energy = deque(maxlen=self._keep_n_episodes)
        self.exploration_not_converged = deque(maxlen=self._keep_n_episodes)
        self.exploration_threshold_exceeded_pct = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_number = 0

        self.use_wandb = "wandb" in sys.modules and os.environ.get("WANDB_API_KEY")
        if self.use_wandb:
            wandb.init(project=config.exp_name, save_code=True, name=config.run_id)
        else:
            warnings.warn("Could not configure wandb access.")

    def log(self, metrics):
        metrics["Exploration episodes number"] = self.exploration_episode_number
        for name, d in zip(
            [
                "episode length",
                "episode return",
                "episode final energy",
                "episode final rl energy",
                "episode not converged",
                "episode threshold exceeded pct",
            ],
            [
                self.exploration_episode_lengths,
                self.exploration_episode_returns,
                self.exploration_episode_final_energy,
                self.exploration_episode_final_rl_energy,
                self.exploration_not_converged,
                self.exploration_threshold_exceeded_pct,
            ],
        ):
            metrics[f"Exploration {name}, mean"] = np.mean(d)
            metrics[f"Exploration {name}, std"] = np.std(d)
        with open(self.metrics_file, "a") as out_metrics:
            json.dump(metrics, out_metrics)
            out_metrics.write("\n")

        if self.use_wandb:
            wandb.log(metrics)

    def update_evaluation_statistics(
        self,
        episode_length,
        episode_return,
        episode_final_energy,
        episode_final_rl_energy,
        threshold_exceeded_pct,
        not_converged,
    ):
        self.exploration_episode_number += 1
        self.exploration_episode_lengths.append(episode_length)
        self.exploration_episode_returns.append(episode_return)
        self.exploration_episode_final_energy.append(episode_final_energy)
        self.exploration_episode_final_rl_energy.append(episode_final_rl_energy)
        self.exploration_threshold_exceeded_pct.append(threshold_exceeded_pct)
        self.exploration_not_converged.append(not_converged)
