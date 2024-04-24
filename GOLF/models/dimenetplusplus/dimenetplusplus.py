import torch
from torch import nn
from torch_geometric.nn.models import DimeNetPlusPlus


class DimeNetPlusPlusPotential(nn.Module):
    def __init__(
        self,
        node_latent_dim: int,
        scaler=None,
        dimenet_hidden_channels=128,
        dimenet_num_blocks=4,
        dimenet_int_emb_size=64,
        dimenet_basis_emb_size=8,
        dimenet_out_emb_channels=256,
        dimenet_num_spherical=7,
        dimenet_num_radial=6,
        dimenet_max_num_neighbors=32,
        dimenet_envelope_exponent=5,
        dimenet_num_before_skip=1,
        dimenet_num_after_skip=2,
        dimenet_num_output_layers=3,
        cutoff=5.0,
        do_postprocessing=False,
        dropout=0.0,
    ):
        super().__init__()

        self.node_latent_dim = node_latent_dim
        self.dimenet_hidden_channels = dimenet_hidden_channels
        self.dimenet_num_blocks = dimenet_num_blocks
        self.dimenet_int_emb_size = dimenet_int_emb_size
        self.dimenet_basis_emb_size = dimenet_basis_emb_size
        self.dimenet_out_emb_channels = dimenet_out_emb_channels
        self.dimenet_num_spherical = dimenet_num_spherical
        self.dimenet_num_radial = dimenet_num_radial
        self.dimenet_max_num_neighbors = dimenet_max_num_neighbors
        self.dimenet_envelope_exponent = dimenet_envelope_exponent
        self.dimenet_num_before_skip = dimenet_num_before_skip
        self.dimenet_num_after_skip = dimenet_num_after_skip
        self.dimenet_num_output_layers = dimenet_num_output_layers
        self.cutoff = cutoff
        self.dropout = dropout

        # print(id(self), "class dropout value", self.dropout)

        self.do_postprocessing = do_postprocessing
        self.scaler = scaler

        self.net = DimeNetPlusPlus(
            hidden_channels=self.dimenet_hidden_channels,
            out_channels=self.node_latent_dim,
            num_blocks=self.dimenet_num_blocks,
            int_emb_size=self.dimenet_int_emb_size,
            basis_emb_size=self.dimenet_basis_emb_size,
            out_emb_channels=self.dimenet_out_emb_channels,
            num_spherical=self.dimenet_num_spherical,
            num_radial=self.dimenet_num_radial,
            cutoff=self.cutoff,
            max_num_neighbors=self.dimenet_max_num_neighbors,
            envelope_exponent=self.dimenet_envelope_exponent,
            num_before_skip=self.dimenet_num_before_skip,
            num_after_skip=self.dimenet_num_after_skip,
            num_output_layers=self.dimenet_num_output_layers,
        )

        self.linear_output_size = 1
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.regr_or_cls_nn = nn.Sequential(
            nn.Linear(self.node_latent_dim, self.node_latent_dim),
            nn.SiLU(),
            nn.Linear(self.node_latent_dim, self.node_latent_dim // 2),
            nn.SiLU(),
            nn.Linear(self.node_latent_dim // 2, self.node_latent_dim // 2),
            nn.SiLU(),
            nn.Linear(self.node_latent_dim // 2, self.linear_output_size),
        )

    @torch.enable_grad()
    def forward(self, data):
        pos = data.pos
        atom_z = data.z
        batch_mapping = data.batch
        pos = pos.requires_grad_(True)
        graph_embeddings = self.net(pos=pos, z=atom_z, batch=batch_mapping)

        if self.dropout:
            # print(id(self), self.dropout, "DROPOUT")
            graph_embeddings = self.dropout_layer(graph_embeddings)

        predictions = torch.flatten(self.regr_or_cls_nn(graph_embeddings).contiguous())
        forces = -1 * (
            torch.autograd.grad(
                predictions,
                pos,
                grad_outputs=torch.ones_like(predictions),
                create_graph=self.training,
            )[0]
        )

        if self.scaler and self.do_postprocessing:
            predictions = self.scaler["scale_"] * predictions + self.scaler["mean_"]

        return predictions, forces
