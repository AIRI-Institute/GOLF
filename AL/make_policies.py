import schnetpack

from AL import DEVICE
from AL.AL_actor import ALPolicy, ALMultiThreadingPolicy, ALAsyncPolicy, ALMultiProcessingPolicy
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
    if args.policy == 'base':
        policy = ALPolicy(
            n_parallel=args.n_parallel,
            backbone=args.backbone,
            backbone_args=backbone_args,
            action_scale_scheduler=args.action_scale_scheduler,
            action_scale=args.action_scale,
            action_norm_limit=args.action_norm_limit,
            max_iter=args.max_iter,
            lbfgs_device=args.lbfgs_device,
        ).to(DEVICE)
    elif args.policy == 'mt':
        policy = ALMultiThreadingPolicy(
            n_parallel=args.n_parallel,
            backbone=args.backbone,
            backbone_args=backbone_args,
            action_scale_scheduler=args.action_scale_scheduler,
            action_scale=args.action_scale,
            action_norm_limit=args.action_norm_limit,
            max_iter=args.max_iter,
            lbfgs_device=args.lbfgs_device,
        ).to(DEVICE)
    elif args.policy == 'async':
        policy = ALAsyncPolicy(
            n_parallel=args.n_parallel,
            backbone=args.backbone,
            backbone_args=backbone_args,
            action_scale_scheduler=args.action_scale_scheduler,
            action_scale=args.action_scale,
            action_norm_limit=args.action_norm_limit,
            max_iter=args.max_iter,
            lbfgs_device=args.lbfgs_device,
        ).to(DEVICE)
    elif args.policy == 'mp':
        policy = ALMultiProcessingPolicy(
            n_parallel=args.n_parallel,
            backbone=args.backbone,
            backbone_args=backbone_args,
            action_scale_scheduler=args.action_scale_scheduler,
            action_scale=args.action_scale,
            action_norm_limit=args.action_norm_limit,
            max_iter=args.max_iter,
            lbfgs_device=args.lbfgs_device,
        ).to(DEVICE)
    else:
        assert False, f'Cannot be here!'

    # Initialize eval policy
    if args.reward == "rdkit":
        n_parallel_eval = 1
    else:
        n_parallel_eval = args.n_eval_runs

    if args.policy == 'base':
        eval_policy = ALPolicy(
            n_parallel=n_parallel_eval,
            backbone=args.backbone,
            backbone_args=backbone_args,
            action_scale_scheduler=args.action_scale_scheduler,
            action_scale=args.action_scale,
            action_norm_limit=args.action_norm_limit,
            max_iter=args.max_iter,
            lbfgs_device=args.lbfgs_device,
            grad_threshold=args.grad_threshold,
        ).to(DEVICE)
    elif args.policy == 'mt':
        eval_policy = ALMultiThreadingPolicy(
            n_parallel=n_parallel_eval,
            backbone=args.backbone,
            backbone_args=backbone_args,
            action_scale_scheduler=args.action_scale_scheduler,
            action_scale=args.action_scale,
            action_norm_limit=args.action_norm_limit,
            max_iter=args.max_iter,
            lbfgs_device=args.lbfgs_device,
            grad_threshold=args.grad_threshold,
        ).to(DEVICE)
    elif args.policy == 'async':
        eval_policy = ALAsyncPolicy(
            n_parallel=n_parallel_eval,
            backbone=args.backbone,
            backbone_args=backbone_args,
            action_scale_scheduler=args.action_scale_scheduler,
            action_scale=args.action_scale,
            action_norm_limit=args.action_norm_limit,
            max_iter=args.max_iter,
            lbfgs_device=args.lbfgs_device,
            grad_threshold=args.grad_threshold,
        ).to(DEVICE)
    elif args.policy == 'mp':
        eval_policy = ALMultiProcessingPolicy(
            n_parallel=n_parallel_eval,
            backbone=args.backbone,
            backbone_args=backbone_args,
            action_scale_scheduler=args.action_scale_scheduler,
            action_scale=args.action_scale,
            action_norm_limit=args.action_norm_limit,
            max_iter=args.max_iter,
            lbfgs_device=args.lbfgs_device,
            grad_threshold=args.grad_threshold,
        ).to(DEVICE)
    else:
        assert False, f'Cannot be here!'

    return policy, eval_policy
