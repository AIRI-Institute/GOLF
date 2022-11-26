import torch
from torch.distributions import Normal

LOG_STD_MIN_MAX = (-20, 2)


from rl import DEVICE


class GenerateActionsBlock(torch.nn.Module):
    def __init__(self, out_embedding_size, limit_actions, cutoff_network):
        super().__init__()
        self.out_embedding_size = out_embedding_size
        #assert tanh in ["before_projection", "after_projection"],\
        #    "Variable tanh must take one of two values: {}, {}".format("before_projection", "after_projection")
        # self.tanh = tanh
        self.limit_actions = limit_actions
        self.cutoff_network = cutoff_network

    def forward(self, kv, positions, atoms_mask, action_scale):
        # Mask kv
        kv *= atoms_mask[..., None]
        k_mu, v_mu, actions_log_std = torch.split(kv, [self.out_embedding_size, self.out_embedding_size, 1], dim=-1)
        
        # Calculate mean and std of shifts relative to other atoms
        # Divide by \sqrt(emb_size) to bring initial action means closer to 0
        rel_shifts_mean = torch.matmul(k_mu, v_mu.transpose(1, 2)) / torch.sqrt(torch.FloatTensor([k_mu.size(-1)])).to(DEVICE)

        # Calculate matrix of 1-vectors to other atoms
        P = positions[:, None, :, :] - positions[:, :, None, :]
        r_ij = torch.norm(P, p=2, dim=-1)
        P /= (r_ij[..., None] + 1e-8)
        
        # Project actions
        # Atoms are assumed to be affected only by atoms inside te cutoff radius
        fcut = self.cutoff_network(r_ij)
        actions_mean = (P * rel_shifts_mean[..., None] * fcut[..., None]).sum(-2)
        
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
            actions = self.scaled_normal.rsample()
            log_prob = self.scaled_normal.log_prob(actions)
            log_prob *= atoms_mask[..., None]
            log_prob = log_prob.sum(dim=(1, 2)).unsqueeze(-1)
        else:
            actions = action_scale * actions_mean
            log_prob = None
        actions *= atoms_mask[..., None]

        return actions, log_prob