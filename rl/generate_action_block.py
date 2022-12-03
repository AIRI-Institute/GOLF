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

        # Matrix X might have negative elements. 
        # A negative x_ij means that atoms have to change places with each other.
        self.activation_x = activation
        self.linear_x = nn.Linear(out_embedding_size, out_embedding_size)
        
        # Matrix K which represents stifness of the string
        # must have positive elements only.
        self.activation_k = nn.ReLU()
        self.linear_k = nn.Linear(out_embedding_size, out_embedding_size)

    def forward(self, kx, positions, atoms_mask, action_scale, eval_actions=None):
        # Mask kx
        kx *= atoms_mask[..., None]

        # Calculate matrix of directions from atoms to all other atoms
        P = positions[:, None, :, :] - positions[:, :, None, :]
        r_ij = torch.norm(P, p=2, dim=-1)
        P /= r_ij[..., None] + 1e-8

        # We model interaction between atoms as springs with stiffness 'k'.
        # First the agent predictes desired distances between atoms.
        # Then we subtract current distances from the predicted distance to obtain x.
        # x and k must be symmetrical to agree with physics.
        x, k, actions_log_std = torch.split(kx, [self.out_embedding_size, self.out_embedding_size, 1], dim=-1)
        
        # TODO check if needed.
        # Right now both x and k linearly depend on backbone's output.
        # Process them separately ?
        x = self.linear_x(self.activation_x(x))
        k = self.linear_k(self.activation_k(k))
        
        # Calculate desired distances between pairs of atoms
        # Divide by \sqrt(emb_size) to bring initial action means closer to 0
        x = torch.matmul(x, x.transpose(1, 2)) / torch.sqrt(torch.FloatTensor([x.size(-1)])).to(DEVICE)

        # Subtract desired distances from current distances between
        # atoms to get delta(x). Divide by two to account for x_ij = x_ji.
        x = (r_ij - x) / 2

        # Calculate stifness 'k'
        # Divide by \sqrt(emb_size) to bring initial action means closer to 0
        k = torch.matmul(k, k.transpose(1, 2)) / torch.sqrt(torch.FloatTensor([k.size(-1)])).to(DEVICE)

        # Get forces
        f = x * k
        
        # Create neighbors mask to cope with padded molecules
        nbh_mask = self.get_nbh_mask(atoms_mask)

        # Atoms are assumed to be affected only by atoms inside the cutoff radius
        fcut = self.cutoff_network(r_ij) * nbh_mask
        
        # Project actions
        # TODO Summation dim must be 'from'. Check it.
        actions_mean = (P * f[..., None] * fcut[..., None]).sum(self.summation_dim)
        
        # # Make actions norm independent of the number of atoms inside cutoff
        # actions_mean /= fcut.sum(self.summation_dim)[..., None] + 1e-8
        
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

    def get_nbh_mask(self, atoms_mask):
        bs, n = atoms_mask.shape
        num_atoms = atoms_mask.sum(-1).long()
        
        nbh_mask = torch.zeros(bs, n, n).to(DEVICE)
        for i in range(bs):
            nbh_mask[i, :num_atoms[i], :num_atoms[i]] = 1.0

        return nbh_mask
