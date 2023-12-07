import yaml
from torch.optim import SGD, Adam

from GOLF import DEVICE
from GOLF.GOLF_actor import (
    Actor,
    ConformationOptimizer,
    LBFGSConformationOptimizer,
    RdkitActor,
)
from GOLF.optim.lion_pytorch import Lion
from utils.utils import ignore_extra_args

actors = {
    "GOLF": ignore_extra_args(Actor),
    "rdkit": ignore_extra_args(RdkitActor),
}


def make_policies(env, eval_env, args):
    # Backbone args
    with open(args.nnp_config_path, "r") as f:
        nnp_args = yaml.safe_load(f)

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
        "lr_scheduler": args.conf_opt_lr_scheduler,
        "t_max": args.timelimit_train,
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

    # Initialize eval policy
    if args.reward == "rdkit":
        n_parallel_eval = 1
    else:
        n_parallel_eval = args.n_eval_runs

    # Update arguments and initialize new actor
    policy_args["n_parallel"] = n_parallel_eval

    actor_args.update({"env": eval_env})
    eval_actor = actors[args.actor](**actor_args)

    if args.conformation_optimizer == "LBFGS":
        eval_policy = ignore_extra_args(LBFGSConformationOptimizer)(
            actor=eval_actor, **policy_args
        ).to(DEVICE)
    else:
        eval_policy = ignore_extra_args(ConformationOptimizer)(
            actor=eval_actor, **policy_args
        ).to(DEVICE)

    return policy, eval_policy
