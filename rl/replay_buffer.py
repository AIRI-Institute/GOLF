import torch
import numpy as np

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from schnetpack.data.loader import _collate_aseatoms

class ReplayBufferPPO(object):
    def __init__(self, device, max_size=int(1e6)):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = [None] * self.max_size
        self.next_states = [None] * self.max_size
        self.actions = [None] * self.max_size
        self.reward = torch.empty((max_size, 1), dtype=torch.float32)
        self.actions_log_probs = torch.empty((max_size, 1), dtype=torch.float32)
        self.not_done = torch.empty((max_size, 1), dtype=torch.float32)
        self.not_ep_end = torch.empty((max_size, 1), dtype=torch.float32)
        self.values = torch.empty((max_size, 1), dtype=torch.float32)
        self.returns = torch.zeros((max_size + 1, 1), dtype=torch.float32)

    def add(self, state, action, next_state, reward, done, ep_end, action_log_prob, value):
        action, reward, done, ep_end, action_log_prob, value = torch.FloatTensor(action),\
                                                               torch.FloatTensor([reward]), \
                                                               torch.FloatTensor([done]), \
                                                               torch.FloatTensor([ep_end]), \
                                                               torch.FloatTensor(action_log_prob), \
                                                               torch.FloatTensor(value)
                               

        self.states[self.ptr] = {k:v.squeeze(0).detach().cpu() for k, v in state.items() if k != "representation"}
        self.next_states[self.ptr] = {k:v.squeeze(0).detach().cpu() for k, v in next_state.items() if k != "representation"}
        self.actions[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.not_ep_end[self.ptr] = 1. - ep_end
        self.actions_log_probs[self.ptr] = action_log_prob
        self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        batch_size = self.max_size

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
            states = [self.states[i] for i in indices]
            state_batch = {k:v.to(self.device) for k, v in _collate_aseatoms(states).items()}
            actions = [self.actions[i] for i in indices]
            action_batch = _collate_actions(actions).to(self.device)
            value_preds_batch = self.values[indices].to(self.device)
            return_batch = self.returns[indices].to(self.device)
            old_action_log_probs_batch = self.actions_log_probs[indices].to(self.device)
            
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].to(self.device)

            yield state_batch, action_batch, value_preds_batch,\
                  return_batch, old_action_log_probs_batch, adv_targ

    def compute_returns(self, next_value, gamma, done_on_timelimit):
        self.returns[-1] = next_value
        if done_on_timelimit:
            # Timeout is considered a done
            for step in reversed(range(self.reward.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.not_done[step] + self.reward[step]
        else:
            # If episode ends with a timeout bootstrap value target
            for step in reversed(range(self.reward.size(0))):
                self.returns[step] = (self.returns[step + 1] * gamma * self.not_done[step] + self.reward[step]) * self.not_ep_end[step] \
                    + (1 - self.not_ep_end[step]) * self.values[step]


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

        num_atoms = states['_atom_mask'].sum(-1).long()
        # Unpad states, next_states and actions
        actions_list = [action[:num_atoms[i]] for i, action in enumerate(actions)]
        states_list = [{k:v[i, :num_atoms[i]] for k, v in states.items()} for i in range(len(num_atoms))]
        next_states_list = [{k:v[i, :num_atoms[i]] for k, v in next_states.items()} for i in range(len(num_atoms))]

        # Update replay buffer
        for i in range(len(num_atoms)):
            self.states[self.ptr] = states_list[i]
            self.next_states[self.ptr] = next_states_list[i]
            self.actions[self.ptr] = actions_list[i]
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


def _collate_actions(actions):
    max_size = max([action.shape[0] for action in actions])
    actions_batch = torch.zeros(len(actions), max_size, actions[0].shape[1])
    for i, action in enumerate(actions):
        actions_batch[i, slice(0, action.shape[0])] = action
    return actions_batch
