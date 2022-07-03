import torch
import numpy as np


class ReplayBuffer(object):
    numpy_to_torch_dtype_dict = {
        np.bool       : torch.bool,
        np.int64      : torch.int64,
        np.float32    : torch.float32,
    }

    def __init__(self, state_dict_names, state_dict_dtypes, state_dims, action_dim, device, max_size=int(1e6)):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state_dict_names = state_dict_names

        # State dict names sorted in alphabetic order
        state_names = ['state' + name for name in state_dict_names]
        next_state_names = ['next_state' + name for name in state_dict_names]

        self.transition_names = (*state_names, 'action', *next_state_names, 'reward', 'not_done')
        sizes = (*state_dims, action_dim, *state_dims, [1], [1])
        torch_state_dict_dtypes = [self.numpy_to_torch_dtype_dict[t.type] for t in state_dict_dtypes]
        dtypes = (*torch_state_dict_dtypes, torch.float32, *torch_state_dict_dtypes, torch.float32, torch.float32)
        for name, size, dtype in zip(self.transition_names, sizes, dtypes):
            setattr(self, name, torch.empty((max_size, *size), dtype=dtype))

    def add(self, state, action, next_state, reward, done):
        # Convert action to torch tensor for Critic
        action, reward, done = torch.FloatTensor(action),\
                               torch.FloatTensor([reward]), \
                               torch.FloatTensor([done])

        sorted_state_values = [state[key].squeeze() for key in sorted(state) if key != "representation"]
        sorted_next_state_values = [next_state[key].squeeze() for key in sorted(next_state) if key != "representation"]
        values = (*sorted_state_values, action, *sorted_next_state_values, reward, 1. - done)
        for name, value in zip(self.transition_names, values):
            getattr(self, name)[self.ptr] = value.detach().cpu()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.choice(self.size, batch_size, replace=False)
        state_dict = {name: getattr(self, 'state' + name)[ind].to(self.device) for name in self.state_dict_names}
        next_state_dict = {name: getattr(self, 'next_state' + name)[ind].to(self.device) for name in self.state_dict_names}
        action, reward, not_done = [getattr(self, name)[ind].to(self.device) for name in ('action', 'reward', 'not_done')]
        return (state_dict, action, next_state_dict, reward, not_done)
