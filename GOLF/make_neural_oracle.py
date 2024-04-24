import copy

import torch
import yaml

from GOLF import DEVICE
from GOLF.make_policies import actors


def make_neural_oracle(actor, args):
    # Initialize a random actor
    with open(args.nnp_config_path, "r") as f:
        nnp_args = yaml.safe_load(f)
    if args.neural_oracle_dropout:
        assert (
            args.nnp_type == "DimenetPlusPlus"
        ), "Dropout is currently implemented only in DimenetPlusPlus NNP"
        nnp_args["dropout"] = args.neural_oracle_dropout
    actor_args = {
        "nnp_type": args.nnp_type,
        "nnp_args": nnp_args,
        "force_norm_limit": args.forces_norm_limit,
    }

    # Load pretrained actor
    neural_oracle = actors[args.actor](**actor_args).to(DEVICE)
    print("Neural oracle id", id(neural_oracle.nnp))
    if args.load_baseline:
        neural_oracle.load_state_dict(
            torch.load(f"{args.load_baseline}_actor", map_location=DEVICE)
        )
    # if args.fixed_neural_oracle_baseline and args.surrogate_oracle_type == "neural":
    #     neural_oracle.load_state_dict(
    #         torch.load(
    #             f"{args.fixed_neural_oracle_baseline}_actor", map_location=DEVICE
    #         )
    #     )

    # Add random components to the weights neural oracle
    # so that it is slightly different from actor
    neural_oracle_state_dict = neural_oracle.state_dict()
    random_actor = actors[args.actor](**actor_args).to(DEVICE)
    random_actor_state_dict = random_actor.state_dict()
    for key in neural_oracle_state_dict:
        neural_oracle_state_dict[key] = (
            1 - args.initial_tau
        ) * random_actor_state_dict[key] + args.initial_tau * neural_oracle_state_dict[
            key
        ]
    neural_oracle.load_state_dict(neural_oracle_state_dict)

    return neural_oracle
