import torch.nn as nn
import schnetpack as spk
import schnetpack.nn as snn

from rl.backbones.painn import PaiNN
from rl.generate_action_block import GenerateActionsBlock
from utils.utils import ignore_extra_args


backbones = {
    "schnet": ignore_extra_args(spk.SchNet),
    "painn": ignore_extra_args(PaiNN)
}


class PPOBase(nn.Module):
    def __init__(self, backbone, backbone_args, out_embedding_size, action_scale_scheduler,
                 limit_actions, summation_order):
        super(PPOBase, self).__init__()
        self.action_scale_scheduler = action_scale_scheduler
        self.activation = snn.activations.shifted_softplus

        self.cutoff_network = snn.get_cutoff_by_string('hard')(backbone_args['cutoff'])
        representation = backbones[backbone](**backbone_args)
        # SchNet backbone is shared between actor and critic
        output_modules = [ 
                                spk.atomistic.Atomwise(
                                    n_in=representation.n_atom_basis,
                                    n_out=out_embedding_size,
                                    n_neurons=[out_embedding_size],
                                    activation=self.activation,
                                    contributions='embedding'
                                )
                            ]
        self.model = spk.atomistic.model.AtomisticModel(representation, output_modules)

        self.linear_embedding_to_kv = nn.Linear(out_embedding_size, 2 * out_embedding_size + 1)
        self.linear_embedding_to_V = nn.Linear(out_embedding_size, 1)
        self.generate_actions_block = GenerateActionsBlock(out_embedding_size, limit_actions,
                                                           self.cutoff_network, summation_order)
    
    def forward(self, state_dict, eval_actions=None):
        action_scale = self.action_scale_scheduler.get_action_scale()
        atoms_mask = state_dict['_atom_mask']
        
        # Get molecule embeddings
        molecule_emb = self.model(state_dict)

        # Individual embedding for each atom
        embedding_for_actor = molecule_emb['embedding']

        # Aggregated embedding for the molecule
        embedding_for_critic = molecule_emb['y']

        # Get actions
        kv = self.linear_embedding_to_kv(self.activation(embedding_for_actor))
        actions, log_prob = self.generate_actions_block(kv, state_dict['_positions'], atoms_mask, action_scale, eval_actions)

        # Get values
        value = self.linear_embedding_to_V(self.activation(embedding_for_critic))
        return value, actions, log_prob

    def select_action(self, state_dict):
        _, actions, _ = self.forward(state_dict)
        return actions.cpu().detach().numpy()


class PPOPolicy(nn.Module):
    def __init__(self, backbone, backbone_args, out_embedding_size,
                 action_scale_scheduler, limit_actions, summation_order):
        super().__init__()
        self.base = PPOBase(backbone, backbone_args, out_embedding_size,
                            action_scale_scheduler, limit_actions, summation_order)

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