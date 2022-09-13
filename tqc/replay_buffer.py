import torch
import numpy as np

from schnetpack.data.loader import _collate_aseatoms

class ReplayBuffer(object):
    numpy_to_torch_dtype_dict = {
        np.bool       : torch.bool,
        np.int64      : torch.int64,
        np.float32    : torch.float32,
    }

    def __init__(self, device, max_size=int(1e6), ppo=False):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.ppo = ppo

        self.states = [None] * self.max_size
        self.next_states = [None] * self.max_size
        self.actions = [None] * self.max_size
        self.reward = torch.empty((max_size, 1), dtype=torch.float32)
        self.not_done = torch.empty((max_size, 1), dtype=torch.float32)
        if ppo:
            self.actions_log_probs = torch.empty((max_size, 1), dtype=torch.float32)
            self.not_done_tl = torch.empty((max_size, 1), dtype=torch.float32)
            self.values = torch.empty((max_size, 1), dtype=torch.float32)
            self.next_values = torch.zeros((max_size, 1), dtype=torch.float32)
            self.returns = torch.zeros((max_size, 1), dtype=torch.float32)

    def add(self, state, action, next_state, reward, done, done_tl=None, action_log_prob=None, 
            value=None, next_value=0):
        # Convert action to torch tensor for Critic
        action, reward, done = torch.FloatTensor(action),\
                               torch.FloatTensor([reward]), \
                               torch.FloatTensor([done])

        self.states[self.ptr] = {k:v.squeeze(0).detach().cpu() for k, v in state.items() if k != "representation"}
        self.next_states[self.ptr] = {k:v.squeeze(0).detach().cpu() for k, v in next_state.items() if k != "representation"}
        self.actions[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        if self.ppo:
            self.not_done_tl[self.ptr] = 1. - done_tl
            self.actions_log_probs[self.ptr] = action_log_prob
            self.values[self.ptr] = value
            self.next_values[self.ptr] = next_value
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.choice(self.size, batch_size, replace=False)
        states = [self.states[i] for i in ind]
        state_batch = {k:v.to(self.device) for k, v in _collate_aseatoms(states).items()}
        next_states = [self.next_states[i] for i in ind]
        next_state_batch = {k:v.to(self.device) for k, v in _collate_aseatoms(next_states).items()}
        actions = [self.actions[i] for i in ind]
        # State batch must include atomic counts to calculate target entropy
        action_batch, atoms_count, mask = [tensor.to(self.device) for tensor in _collate_actions(actions)]
        state_batch.update({
            '_atoms_count': atoms_count,
            '_atoms_mask': mask
        })
        next_state_batch.update({
            '_atoms_count': atoms_count,
            '_atoms_mask': mask
        })
        reward, not_done = [getattr(self, name)[ind].to(self.device) for name in ('reward', 'not_done')]
        tuple_ = [state_batch, action_batch, next_state_batch, reward, not_done]
        if self.ppo:
            actions_log_probs, values, returns = [getattr(self, name)[ind].to(self.device) for name \
                in ('actions_log_probs', 'values', 'returns')]
            tuple_.extend([actions_log_probs, values, returns])
        return tuple_
    
    def compute_returns(self, gamma):
        assert self.size == self.max_size, 'compute_returns work only for filled buffer'
        step = -1
        self.returns[step] = (self.next_values[step] * \
            gamma * self.not_done[step] + self.reward[step]) * self.not_done_tl[step] \
            + (1 - self.not_done_tl[step]) * (gamma * self.next_values[step] + self.reward[step])

        for step in reversed(range(self.reward.size(0) - 1)):
            self.returns[step] = (self.returns[step + 1] * \
                gamma * self.not_done[step] + self.reward[step]) * self.not_done_tl[step] \
                + (1 - self.not_done_tl[step]) * (gamma * self.next_values[step] + self.reward[step])


def _collate_actions(actions):
    atoms_count = []
    max_size = max([action.shape[0] for action in actions])
    actions_batch = torch.zeros(len(actions), max_size, actions[0].shape[1])
    for i, action in enumerate(actions):
        atoms_count.append(action.shape[0])
        actions_batch[i, slice(0, action.shape[0])] = action
    atoms_count = torch.LongTensor(atoms_count)
    # Create action mask for critic
    mask = torch.arange(max_size).expand(len(atoms_count), max_size) < atoms_count.unsqueeze(1)
    return actions_batch, atoms_count, mask
