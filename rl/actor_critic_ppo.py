import copy
import torch
import torch.nn as nn
import schnetpack as spk

from torch.distributions import Normal

from rl import DEVICE


LOG_STD_MIN_MAX = (-20, 2)


class GenerateActionsBlock(nn.Module):
    def __init__(self, out_embedding_size, tanh):
        super().__init__()
        self.out_embedding_size = out_embedding_size
        assert tanh in ["before_projection", "after_projection"],\
            "Variable tanh must take one of two values: {}, {}".format("before_projection", "after_projection")
        self.tanh = tanh

    def forward(self, kv, positions, atoms_mask, action_scale, eval_actions=None):
        # Mask kv
        kv *= atoms_mask[..., None]
        k_mu, v_mu, actions_log_std = torch.split(kv, [self.out_embedding_size, self.out_embedding_size, 1], dim=-1)
        
        # Calculate mean and std of shifts relative to other atoms
        # Divide by \sqrt(emb_size) to bring initial action means closer to 0
        rel_shifts_mean = torch.matmul(k_mu, v_mu.transpose(1, 2)) / torch.sqrt(torch.FloatTensor([k_mu.size(-1)])).to(DEVICE)
        
        # Bound relative_shifts with tanh if self.tanh == "before_projection"
        if  self.tanh == "before_projection":
            rel_shifts_mean = torch.tanh(rel_shifts_mean)
        
        # Calculate matrix of 1-vectors to other atoms
        P = positions[:, :, None, :] - positions[:, None, :, :]
        norm = torch.norm(P, p=2, dim=-1) + 1e-8
        P /= norm[..., None]
        
        # Project actions
        actions_mean = (P * rel_shifts_mean[..., None]).sum(-2)
        
        # Make actions norm independent of the number of atoms
        actions_mean /= atoms_mask.sum(-1)[:, None, None]
        
        # Bound means with tanh if self.tanh == "after_projection"
        if self.tanh == "after_projection":
            actions_mean = torch.tanh(actions_mean)

        print("Action means shape = ", actions_mean.shape)
        norm_reshape = torch.norm(actions_mean * action_scale, p=2, dim=-1).reshape(1, -1).squeeze()
        mask_reshape = atoms_mask.reshape(1, -1).squeeze().long()
        print("training = ", self.training, " norm = ", norm_reshape[mask_reshape].mean())

        if self.training:
            # Clamp and exp log_std
            actions_log_std = actions_log_std.clamp(*LOG_STD_MIN_MAX)
            actions_std = torch.exp(actions_log_std)
            # Sample actions and calculate log prob
            self.scaled_normal = Normal(actions_mean * action_scale, actions_std * action_scale)
            if eval_actions is None:
                actions = self.scaled_normal.sample() # think about rsample/sample
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

class PPOBase(nn.Module):
    def __init__(self, schnet_args, out_embedding_size, action_scale_scheduler, tanh="after_projection"):
        super(PPOBase, self).__init__()
        self.action_scale_scheduler = action_scale_scheduler
        self.out_embedding_size = out_embedding_size
        # SchNet backbone is shared between actor and critic
        schnet = spk.SchNet(
                        n_interactions=schnet_args["n_interactions"], #3
                        cutoff=schnet_args["cutoff"], #20.0
                        n_gaussians=schnet_args["n_gaussians"] #50
                    )
        output_modules = [ 
                                spk.atomistic.Atomwise(
                                    n_in=schnet.n_atom_basis,
                                    n_out=out_embedding_size,
                                    n_neurons=[out_embedding_size],
                                    contributions='embedding'
                                )
                            ]
        self.model = spk.atomistic.model.AtomisticModel(schnet, output_modules)

        self.linear_embedding_to_kv = nn.Linear(out_embedding_size, 2 * out_embedding_size + 1)
        self.linear_embedding_to_V = nn.Linear(out_embedding_size, 1)
        self.generate_actions_block = GenerateActionsBlock(out_embedding_size, tanh)
    
    def forward(self, state_dict, eval_actions=None):
        action_scale = self.action_scale_scheduler.get_action_scale()
        if '_atoms_mask' not in state_dict:
            atoms_mask = torch.ones(state_dict['_positions'].shape[:2]).to(DEVICE)
        else:
            atoms_mask = state_dict['_atoms_mask']
        
        # Get embedding
        schnet_out = self.model(state_dict)
        embedding_for_actor = schnet_out['embedding']
        embedding_for_critic = schnet_out['y']

        # Get actions
        kv = self.linear_embedding_to_kv(embedding_for_actor)
        actions, log_prob = self.generate_actions_block(kv, state_dict['_positions'], atoms_mask, action_scale, eval_actions)

        # Get values
        value = self.linear_embedding_to_V(embedding_for_critic)
        return value, actions, log_prob

    def select_action(self, state_dict):
        action, _, _ = self.forward(state_dict)
        return action[0].cpu().detach().numpy()


class PPOPolicy(nn.Module):
    def __init__(self, schnet_args, out_embedding_size, action_scale_scheduler, tanh="after_projection"):
        super().__init__()
        self.base = PPOBase(schnet_args, out_embedding_size, action_scale_scheduler, tanh).to(DEVICE)

    def act(self, state_dict):
        value, action, log_prob = self.base(state_dict)
        return value, action, log_prob

    def get_value(self, state_dict):
        value, _, _ = self.base(state_dict)
        return value 

    def evaluate_actions(self, state_dict, eval_actions):
        value, _, log_prob = self.base(state_dict, eval_actions)
        return value, log_prob
        
    def select_action(self, state_dict):
        return self.base.select_action(state_dict)