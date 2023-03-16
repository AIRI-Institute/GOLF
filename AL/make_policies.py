import schnetpack

from AL import DEVICE
from AL.AL_actor import ALPolicy
from AL.utils import get_cutoff_by_string


def make_policies(args):
    # Backbone args
    backbone_args = {
        "n_interactions": args.n_interactions,
        "n_atom_basis": args.n_atom_basis,
        "radial_basis": schnetpack.nn.BesselRBF(n_rbf=args.n_rbf, cutoff=args.cutoff),
        "cutoff_fn": get_cutoff_by_string("cosine")(args.cutoff),
    }

    # Initialize policy
    policy = ALPolicy(
        n_parallel=args.n_parallel,
        backbone=args.backbone,
        backbone_args=backbone_args,
        action_scale_scheduler=args.action_scale_scheduler,
        action_scale=args.action_scale,
        action_norm_limit=args.action_norm_limit,
        max_iter=args.max_iter,
    ).to(DEVICE)

    # Initialize eval policy
    if args.reward == "rdkit":
        n_parallel_eval = 1
    else:
        n_parallel_eval = args.n_eval_runs
    eval_policy = ALPolicy(
        n_parallel=n_parallel_eval,
        backbone=args.backbone,
        backbone_args=backbone_args,
        action_scale_scheduler=args.action_scale_scheduler,
        action_scale=args.action_scale,
        action_norm_limit=args.action_norm_limit,
        max_iter=args.max_iter,
    ).to(DEVICE)

    return policy, eval_policy
