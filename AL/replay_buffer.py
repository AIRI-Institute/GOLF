import torch
import numpy as np

from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.nn import scatter_add

from AL.utils import unpad_state
from env.moldynamics_env import env_fn
from env.wrappers import RewardWrapper


NORM_THRESHOLD = 10.5


class ReplayBuffer(object):
    def __init__(
        self,
        device,
        max_size=int(1e6),
        atomrefs=None,
        initial_RB=None,
        initial_conf_pct=0.0,
    ):
        self.device = device
        self.max_size = max_size
        self.initial_RB = initial_RB
        self.ptr = 0
        self.size = 0
        self.replay_buffer_full = False

        if self.initial_RB:
            self.initial_conf_pct = initial_conf_pct
        else:
            self.initial_conf_pct = 0.0

        self.states = [None] * self.max_size
        self.energy = torch.empty((max_size, 1), dtype=torch.float32)
        self.forces = [None] * self.max_size

        if atomrefs:
            self.atomrefs = torch.tensor(atomrefs, device=device)
        else:
            self.atomrefs = None

    def add(self, states, forces, energies):
        energies = torch.tensor(energies, dtype=torch.float32)
        force_norms = np.array([np.linalg.norm(force) for force in forces])
        individual_states = unpad_state(states)
        # Exclude conformations with forces that have a high norm
        # from the replay buffer
        for i in np.where(force_norms < NORM_THRESHOLD)[0]:
            self.states[self.ptr] = individual_states[i]
            self.energy[self.ptr] = energies[i]
            self.forces[self.ptr] = torch.tensor(forces[i], dtype=torch.float32)
            self.ptr = (self.ptr + 1) % self.max_size
            # self.size = min(self.size + 1, self.max_size)
            self.size = self.size + 1

        self.replay_buffer_full = self.size >= self.max_size

    def sample(self, batch_size):
        new_samples_batch_size = int(batch_size * (1 - self.initial_conf_pct))
        states, forces, energy = self.sample_wo_collate(new_samples_batch_size)

        if self.initial_RB and self.initial_conf_pct:
            initial_conf_batch_size = batch_size - new_samples_batch_size
            init_states, init_forces, init_energy = self.initial_RB.sample_wo_collate(
                initial_conf_batch_size
            )
            states = states + init_states
            forces = forces + init_forces
            energy = torch.cat((energy, init_energy), dim=0)

        state_batch = {
            key: value.to(self.device)
            for key, value in _atoms_collate_fn(states).items()
        }
        forces = torch.cat(forces).to(self.device)

        if self.atomrefs is not None:
            # Get system index
            idx_m = state_batch[properties.idx_m]

            # Get num molecules in the batch
            max_m = int(idx_m[-1]) + 1

            # Get atomization energy for each molecule in the batch
            atomization_energy = scatter_add(
                self.atomrefs[state_batch[properties.Z]], idx_m, dim_size=max_m
            ).unsqueeze(-1)
            energy -= atomization_energy

        return state_batch, forces, energy

    def sample_wo_collate(self, batch_size):
        ind = np.random.choice(self.size, batch_size, replace=False)
        states = [self.states[i] for i in ind]
        forces = [self.forces[i] for i in ind]
        energy = self.energy[ind].to(self.device)
        return states, forces, energy


def fill_initial_replay_buffer(device, args, atomrefs=None):
    # Env kwargs
    env_kwargs = {
        "db_path": args.db_path,
        "n_parallel": 1,
        "timelimit": args.timelimit_train,
        "sample_initial_conformations": False,
        "num_initial_conformations": args.num_initial_conformations,
    }
    # Initialize env
    env = env_fn(**env_kwargs)
    if args.num_initial_conformations == -1:
        total_confs = env.get_db_length()
    else:
        total_confs = args.num_initial_conformations

    # Reward wrapper kwargs
    reward_wrapper_kwargs = {
        "dft": args.reward == "dft",
        "n_threads": args.n_threads,
        "minimize_on_every_step": args.minimize_on_every_step,
        "molecules_xyz_prefix": args.molecules_xyz_prefix,
        "terminate_on_negative_reward": args.terminate_on_negative_reward,
        "max_num_negative_rewards": args.max_num_negative_rewards,
    }
    # Initialize reward wrapper
    env = RewardWrapper(env, **reward_wrapper_kwargs)

    initial_replay_buffer = ReplayBuffer(
        device, max_size=total_confs, atomrefs=atomrefs
    )

    # Fill up the replay buffer
    # for _ in range(100):
    for _ in range(total_confs):
        state = env.reset()
        # Save initial state in replay buffer
        energies = env.get_energies()
        forces = env.get_forces()
        initial_replay_buffer.add(state, forces, energies)

    return initial_replay_buffer
