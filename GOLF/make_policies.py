import yaml
from torch.optim import SGD, Adam

from GOLF import DEVICE
from GOLF.GOLF_actor import Actor, RdkitActor
from GOLF.optim import Lion, ConformationOptimizer, LBFGSConformationOptimizer
from utils.utils import ignore_extra_args

actors = {
    "GOLF": ignore_extra_args(Actor),
    "rdkit": ignore_extra_args(RdkitActor),
}


def make_policies(env, args):
    # Backbone args
    with open(args.nnp_config_path, "r") as f:
        nnp_args = yaml.safe_load(f)

    if args.actor_dropout:
        assert (
            args.nnp_type == "DimenetPlusPlus"
        ), "Dropout is currently implemented only in DimenetPlusPlus NNP"
        nnp_args["dropout"] = args.actor_dropout

    # Actor args
    actor_args = {
        "env": env,
        "nnp_type": args.nnp_type,
        "nnp_args": nnp_args,
        "force_norm_limit": args.forces_norm_limit,
    }
    actor = actors[args.actor](**actor_args)

    policy_args = {
        "n_parallel": args.n_parallel,
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
    elif args.conformation_optimizer == "Adam":
        policy_args.update(
            {"optimizer": Adam, "optimizer_kwargs": {"lr": args.conf_opt_lr}}
        )
    else:
        raise NotImplemented("Unknowm policy type: {}!".format(args.policy))

    if args.conformation_optimizer == "LBFGS":
        policy = ignore_extra_args(LBFGSConformationOptimizer)(
            actor=actor, **policy_args
        ).to(DEVICE)
    else:
        policy = ignore_extra_args(ConformationOptimizer)(
            actor=actor, **policy_args
        ).to(DEVICE)

    return policy
