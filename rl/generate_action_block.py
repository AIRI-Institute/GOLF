import torch
from torch import nn
from torch.distributions import Normal


from rl import DEVICE


LOG_STD_MIN_MAX = (-20, 2)


class GenerateActionsBlock(nn.Module):
    def __init__(self, out_embedding_size, limit_actions, cutoff_network, summation_order, activation):
        super().__init__()
        self.out_embedding_size = out_embedding_size
        self.limit_actions = limit_actions
        self.cutoff_network = cutoff_network

        if summation_order == "to":
            self.summation_dim = 1
        elif summation_order == "from":
            self.summation_dim = 2

        # self.activation = activation
        # self.linear_k = nn.Linear(out_embedding_size, out_embedding_size)
        # self.linear_v = nn.Linear(out_embedding_size, out_embedding_size)

    def forward(self, kv, positions, atoms_mask, action_scale, eval_actions=None):
        # Mask kv
        kv *= atoms_mask[..., None]
        k_mu, v_mu, actions_log_std = torch.split(kv, [self.out_embedding_size, self.out_embedding_size, 1], dim=-1)
        
        # # Right now both k and v linearly depend on backbone's output.
        # # Process them separately
        # k_mu = self.linear_k(self.activation(k_mu))
        # v_mu = self.linear_k(self.activation(v_mu))

        # Calculate mean and std of shifts relative to other atoms
        # Divide by \sqrt(emb_size) to bring initial action means closer to 0
        rel_shifts_mean = torch.matmul(k_mu, v_mu.transpose(1, 2)) / torch.sqrt(torch.FloatTensor([k_mu.size(-1)])).to(DEVICE)

        # Calculate matrix of directions from atoms to all other atoms
        P = positions[:, None, :, :] - positions[:, :, None, :]
        r_ij = torch.norm(P, p=2, dim=-1)
        P /= r_ij[..., None] + 1e-8

        # Project actions
        # Atoms are assumed to be affected only by atoms inside the cutoff radius
        fcut = self.cutoff_network(r_ij)
        actions_mean = (P * rel_shifts_mean[..., None] * fcut[..., None]).sum(self.summation_dim)
        
        # Make actions norm independent of the number of atoms
        actions_mean /= atoms_mask.sum(-1)[:, None, None]
        
        # Limit actions by scaling their norms
        if self.limit_actions:
            actions_norm = torch.norm(actions_mean, p=2, dim=-1) + 1e-8
            actions_mean = (actions_mean / actions_norm[..., None]) * torch.tanh(actions_norm)[..., None]

        if self.training:
            # Clamp and exp log_std
            actions_log_std = actions_log_std.clamp(*LOG_STD_MIN_MAX)
            actions_std = torch.exp(actions_log_std)
            # Sample actions and calculate log prob
            self.scaled_normal = Normal(actions_mean * action_scale, actions_std * action_scale)
            # In case of PPO
            if eval_actions is None:
                actions = self.scaled_normal.rsample() # think about rsample/sample
            else:
                actions = eval_actions
            log_prob = self.scaled_normal.log_prob(actions)
            log_prob *= atoms_mask[..., None]
            log_prob = log_prob.sum(dim=(1, 2)).unsqueeze(-1)
        else:
            actions = action_scale * actions_mean
            log_prob = None
        actions *= atoms_mask[..., None]

        return actions, log_prob
