import schnetpack
from torch.optim import SGD

from AL import DEVICE
from AL.AL_actor import ConformationOptimizer, LBFGSConformationOptimizer
from AL.utils import get_cutoff_by_string
from AL.optim.lion_pytorch import Lion
from utils.utils import ignore_extra_args


def make_policies(args):
    # Backbone args
    backbone_args = {
        "n_interactions": args.n_interactions,
        "n_atom_basis": args.n_atom_basis,
        "radial_basis": schnetpack.nn.BesselRBF(n_rbf=args.n_rbf, cutoff=args.cutoff),
        "cutoff_fn": get_cutoff_by_string("cosine")(args.cutoff),
    }

    policy_args = {
        "n_parallel": args.n_parallel,
        "backbone": args.backbone,
        "backbone_args": backbone_args,
        "lr_scheduler": args.conf_opt_lr_scheduler,
        "t_max": args.timelimit_train,
        "action_norm_limit": args.action_norm_limit,
    }

    if args.conformation_optimizer == "LBFGS":
        policy_args.update(
            {
                "grad_threshold": args.grad_threshold,
                "lbfgs_device": args.lbfgs_device,
                "optimizer_kwargs": {
                    "lr": 1,
                    "max_iter": args.max_iter,
                },
            }
        )
        policy = ignore_extra_args(LBFGSConformationOptimizer)(**policy_args).to(DEVICE)
    elif args.conformation_optimizer == "GD":
        policy_args.update(
            {
                "optimizer": SGD,
                "optimizer_kwargs": {
                    "lr": args.conf_opt_lr,
                    "momentum": args.momentum,
                },
            }
        )
        policy = ignore_extra_args(ConformationOptimizer)(**policy_args).to(DEVICE)
    elif args.conformation_optimizer == "Lion":
        policy_args.update(
            {
                "optimizer": Lion,
                "optimizer_kwargs": {
                    "lr": args.conf_opt_lr,
                    "betas": (args.lion_beta1, args.lion_beta2),
                },
            }
        )
        policy = ignore_extra_args(ConformationOptimizer)(**policy_args).to(DEVICE)
    else:
        raise NotImplemented("Unknowm policy type: {}!".format(args.policy))

    # Initialize eval policy
    if args.reward == "rdkit":
        n_parallel_eval = 1
    else:
        n_parallel_eval = args.n_eval_runs

    # Update argument
    policy_args["n_parallel"] = n_parallel_eval

    if args.conformation_optimizer == "LBFGS":
        eval_policy = ignore_extra_args(LBFGSConformationOptimizer)(**policy_args).to(
            DEVICE
        )
    elif args.conformation_optimizer == "GD" or args.conformation_optimizer == "Lion":
        eval_policy = ignore_extra_args(ConformationOptimizer)(**policy_args).to(DEVICE)
    else:
        raise NotImplemented("Unknowm policy type: {}!".format(args.policy))

    return policy, eval_policy
