import torch
import numpy as np

from schnetpack.data.loader import _collate_aseatoms

class ReplayBuffer(object):
    numpy_to_torch_dtype_dict = {
        np.bool       : torch.bool,
        np.int64      : torch.int64,
        np.float32    : torch.float32,
    }

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

    def add(self, state, action, next_state, reward, done):
        # Convert action to torch tensor for Critic
        action, reward, done = torch.FloatTensor(action),\
                               torch.FloatTensor([reward]), \
                               torch.FloatTensor([done])

        self.states[self.ptr] = {k:v.squeeze(0).detach().cpu() for k, v in state.items() if k != "representation"}
        self.next_states[self.ptr] = {k:v.squeeze(0).detach().cpu() for k, v in next_state.items() if k != "representation"}
        self.actions[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

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


def _collate_actions(actions):
    max_size = max([action.shape[0] for action in actions])
    actions_batch = torch.zeros(len(actions), max_size, actions[0].shape[1])
    for i, action in enumerate(actions):
        actions_batch[i, slice(0, action.shape[0])] = action
    return actions_batch
