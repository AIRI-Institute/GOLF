import torch
import numpy as np

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from schnetpack.data.loader import _collate_aseatoms


UNWANTED_KEYS = ["representation", "vector_representation"]


class ReplayBufferPPO(object):
    def __init__(self, device, n_processes, max_size):
        self.device = device
        self.max_size = max_size
        self.n_processes = n_processes
        self.ptr = 0
        self.size = 0

        self.states = [[None for _ in range(self.n_processes)] for _ in range(self.max_size)]
        self.next_states = [[None for _ in range(self.n_processes)] for _ in range(self.max_size)]
        self.actions = [[None for _ in range(self.n_processes)] for _ in range(self.max_size)]
        self.reward = torch.empty((max_size, n_processes), dtype=torch.float32)
        self.actions_log_probs = torch.empty((max_size, n_processes), dtype=torch.float32)
        self.not_done = torch.empty((max_size, n_processes), dtype=torch.float32)
        self.not_ep_end = torch.empty((max_size, n_processes), dtype=torch.float32)
        self.values = torch.empty((max_size, n_processes), dtype=torch.float32)
        self.returns = torch.zeros((max_size + 1, n_processes), dtype=torch.float32)

    def add(self, states, actions, next_states, rewards, dones, ep_ends, action_log_probs, values):
        actions, rewards, dones, ep_ends, action_log_probs, values = torch.FloatTensor(actions),\
                                                                     torch.FloatTensor(rewards), \
                                                                     torch.FloatTensor(dones), \
                                                                     torch.FloatTensor(ep_ends), \
                                                                     torch.FloatTensor(action_log_probs), \
                                                                     torch.FloatTensor(values)
                               
        # Store transitions
        for proc_num in range(len(rewards)):
            self.states[self.ptr][proc_num] = {k: v[proc_num].cpu() for k, v in states.items() \
                                               if k not in UNWANTED_KEYS}
            self.next_states[self.ptr][proc_num] = {k: v[proc_num].cpu() for k, v in next_states.items()\
                                                    if k not in UNWANTED_KEYS}
            self.actions[self.ptr][proc_num] = actions[proc_num].cpu()
        
        self.reward[self.ptr].copy_(rewards)
        self.not_done[self.ptr].copy_(1. - dones)
        self.not_ep_end[self.ptr].copy_(1. - ep_ends)
        self.actions_log_probs[self.ptr].copy_(action_log_probs)
        self.values[self.ptr].copy_(values)
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        batch_size = self.max_size * self.n_processes

        states_flat = [state for n_proc_state in self.states for state in n_proc_state]
        actions_flat = [action for n_proc_actions in self.actions for action in n_proc_actions]
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes size of RB {}"
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(self.max_size, num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True
        )
        
        for indices in sampler:
            states = [states_flat[i] for i in indices]
            state_batch = {k:v.to(self.device) for k, v in _collate_aseatoms(states).items()}
            actions = [actions_flat[i] for i in indices]
            action_batch = _collate_actions(actions).to(self.device)
            value_preds_batch = self.values.view(-1, 1)[indices].to(self.device)
            return_batch = self.returns[:-1].view(-1, 1)[indices].to(self.device)
            old_action_log_probs_batch = self.actions_log_probs.view(-1, 1)[indices].to(self.device)
            
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices].to(self.device)

            yield state_batch, action_batch, value_preds_batch,\
                  return_batch, old_action_log_probs_batch, adv_targ

    def compute_returns(self, next_value, gamma, done_on_timelimit):
        self.returns[-1] = next_value.squeeze(-1)
        if done_on_timelimit:
            # Timeout is considered a done
            for step in reversed(range(self.reward.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.not_done[step] + self.reward[step]
        else:
            # If episode ends with a timeout bootstrap value target
            for step in reversed(range(self.reward.size(0))):
                self.returns[step] = (self.returns[step + 1] * gamma * self.not_done[step] + self.reward[step]) \
                                     * self.not_ep_end[step] + (1 - self.not_ep_end[step]) * self.values[step]


class ReplayBufferTQC(object):
    def __init__(self, device, max_size=int(1e6)):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = [None] * self.max_size
        self.next_states = [None] * self.max_size
        self.actions = [None] * self.max_size
        self.reward = torch.empty((max_size, 1), dtype=torch.float32)
        self.not_done = torch.empty((max_size, 1), dtype=torch.float32)

    def add(self, states, actions, next_states, rewards, dones):
        # Convert action to torch tensor for Critic
        actions, rewards, dones = torch.FloatTensor(actions),\
                                  torch.FloatTensor(rewards),\
                                  torch.FloatTensor(dones)

        
        # Store transitions
        for i in range(len(rewards)):
            self.states[self.ptr] = {k:v[i].cpu() for k, v in states.items() if k not in UNWANTED_KEYS}
            self.next_states[self.ptr] = {k:v[i].cpu() for k, v in next_states.items() if k not in UNWANTED_KEYS}
            self.actions[self.ptr] = actions[i].cpu()
            self.reward[self.ptr] = rewards[i]
            self.not_done[self.ptr] = 1. - dones[i]
            
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.choice(self.size, batch_size, replace=False)
        states = [self.states[i] for i in ind]
        state_batch = {k:v.to(self.device) for k, v in _collate_aseatoms(states).items()}
        next_states = [self.next_states[i] for i in ind]
        next_state_batch = {k:v.to(self.device) for k, v in _collate_aseatoms(next_states).items()}
        actions = [self.actions[i] for i in ind]
        action_batch = _collate_actions(actions).to(self.device)
        reward, not_done = [getattr(self, name)[ind].to(self.device) for name in ('reward', 'not_done')]
        return (state_batch, action_batch, next_state_batch, reward, not_done)


class ReplayBufferGD(object):
    def __init__(self, device, max_size=int(1e6)):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = [None] * self.max_size
        self.energy = torch.empty((max_size, 1), dtype=torch.float32)

    def add(self, states, actions, next_state, energies, dones):
        energies = torch.tensor(energies, dtype=torch.float32)

        num_atoms = states['_atom_mask'].sum(-1).long()
        states_list = [{key: value[i, :num_atoms[i]].cpu() for key, value in states.items() if key not in UNWANTED_KEYS}
                       for i in range(len(num_atoms))]

        # Update replay buffer
        for i in range(len(num_atoms)):
            self.states[self.ptr] = states_list[i]
            self.energy[self.ptr] = energies[i]
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.choice(self.size, batch_size, replace=False)
        states = [self.states[i] for i in ind]
        state_batch = {key: value.to(self.device) for key, value in _collate_aseatoms(states).items()}
        energy = self.energy[ind].to(self.device)
        return state_batch, energy


def _collate_actions(actions):
    max_size = max([action.shape[0] for action in actions])
    actions_batch = torch.zeros(len(actions), max_size, actions[0].shape[1])
    for i, action in enumerate(actions):
        actions_batch[i, slice(0, action.shape[0])] = action
    return actions_batch
