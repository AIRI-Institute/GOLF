import torch.nn as nn
import schnetpack as spk
import schnetpack.nn as snn

from rl.backbones.painn import PaiNN
from rl.generate_action_block import SpringAndMassAction, DistanceChangeAction
from utils.utils import ignore_extra_args


backbones = {
    "schnet": ignore_extra_args(spk.SchNet),
    "painn": ignore_extra_args(PaiNN)
}

generate_action_block = {
    "delta_x": ignore_extra_args(DistanceChangeAction),
    "spring_and_mass": ignore_extra_args(SpringAndMassAction)
}


class PPOBase(nn.Module):
    def __init__(self, backbone, backbone_args, generate_action_type, out_embedding_size, action_scale,
                 limit_actions, summation_order, cutoff_type, use_activation):
        super(PPOBase, self).__init__()
        self.action_scale = action_scale
        representation = backbones[backbone](**backbone_args)
        output_modules = [
            spk.atomistic.Atomwise(
                n_in=representation.n_atom_basis,
                n_out=out_embedding_size,
                contributions='embedding',
            )
        ]
        self.model = spk.atomistic.model.AtomisticModel(representation, output_modules)

        self.linear_emb_to_atoms_emb = nn.Linear(out_embedding_size, 2 * out_embedding_size + 1)
        self.linear_emb_to_V = nn.Linear(out_embedding_size, 1)
        
        if use_activation or generate_action_type == "spring_and_mass":
            activation = snn.activations.shifted_softplus
        else:
            activation = None
        self.generate_actions_block = generate_action_block[generate_action_type](
            out_embedding_size, limit_actions, action_scale, cutoff_type,
            backbone_args['cutoff'], summation_order, activation
        )
    
    def forward(self, state_dict, eval_actions=None):
        atoms_mask = state_dict['_atom_mask']
        
        # Get molecule embeddings
        molecule_emb = self.model(state_dict)

        # Individual embedding for each atom
        embedding_for_actor = molecule_emb['embedding']

        # Aggregated embedding for the molecule
        embedding_for_critic = molecule_emb['y']

        # Get actions
        atoms_emb = self.linear_emb_to_atoms_emb(self.activation(embedding_for_actor))
        actions, log_prob = self.generate_actions_block(atoms_emb, state_dict['_positions'], atoms_mask, eval_actions)

        # Get values
        value = self.linear_emb_to_V(self.activation(embedding_for_critic))
        return value, actions, log_prob

    def select_action(self, state_dict):
        _, actions, _ = self.forward(state_dict)
        return actions.cpu().detach().numpy()


class PPOPolicy(nn.Module):
    def __init__(self, backbone, backbone_args, generate_action_type, out_embedding_size,
                 action_scale, limit_actions, summation_order, cutoff_type, use_activation):
        super().__init__()
        self.base = PPOBase(backbone, backbone_args, generate_action_type, out_embedding_size,
                            action_scale, limit_actions, summation_order, cutoff_type, use_activation)

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