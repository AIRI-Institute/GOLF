import copy

import torch

from GOLF import DEVICE


def make_neural_oracle(actor, args):
    neural_oracle = copy.deepcopy(actor)
    if args.fixed_neural_oracle_baseline and args.surrogate_oracle_type == "neural":
        neural_oracle.load_state_dict(
            torch.load(
                f"{args.fixed_neural_oracle_baseline}_actor", map_location=DEVICE
            )
        )
    elif args.load_baseline:
        neural_oracle.load_state_dict(
            torch.load(f"{args.load_baseline}_actor", map_location=DEVICE)
        )
    return neural_oracle
