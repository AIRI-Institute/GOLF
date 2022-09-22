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
            
            # State batch must include atomic counts to calculate target entropy
            action_batch, atoms_count, mask = [tensor.to(self.device) for tensor in _collate_actions(actions)]
            state_batch.update({
                '_atoms_count': atoms_count,
                '_atoms_mask': mask
            })
            
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
            print("NOT DONE")
            print(self.not_done[:20])
            print("NOT EP END")
            print(self.not_ep_end[:20])
            print("REWARDS")
            print(self.reward[:20])
            # Timeout is considered a done
            for step in reversed(range(self.reward.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.not_done[step] + self.reward[step]
            print("RETURNS")
            print(self.returns[:20])
        else:
            # If episode ends with a timeout bootstrap value target
            print("NOT DONE")
            print(self.not_done[:20])
            print("NOT EP END")
            print(self.not_ep_end[:20])
            print("REWARDS")
            print(self.reward[:20])
            for step in reversed(range(self.reward.size(0))):
                self.returns[step] = (self.returns[step + 1] * gamma * self.not_done[step] + self.reward[step]) * self.not_ep_end[step] \
                    + (1 - self.not_ep_end[step]) * self.values[step]
            print("RETURNS")
            print(self.returns[:20])


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
        return (state_batch, action_batch, next_state_batch, reward, not_done)


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