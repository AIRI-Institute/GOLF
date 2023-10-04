import argparse


def str2bool(s):
    """helper function used in order to support boolean command line arguments"""
    if s.lower() in ("true", "t", "1"):
        return True
    elif s.lower() in ("false", "f", "0"):
        return False
    else:
        return s


def check_positive(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError(
            f"{int_value} is an invalid positive int value"
        )
    return int_value


def none_or_str(value):
    if value == "None":
        return None
    return value


def get_args():
    parser = argparse.ArgumentParser()

    # Env args
    parser.add_argument(
        "--n_parallel",
        default=1,
        type=int,
        help="Number of copies of env to run in parallel",
    )
    parser.add_argument(
        "--n_threads",
        default=1,
        type=int,
        help="Number of parallel threads for DFT computations",
    )
    parser.add_argument(
        "--db_path",
        default="env/data/malonaldehyde.db",
        type=str,
        help="Path to molecules database for training",
    )
    parser.add_argument(
        "--eval_db_path",
        default="",
        type=str,
        help="Path to molecules database for evaluation",
    )
    parser.add_argument(
        "--num_initial_conformations",
        default=50000,
        type=int,
        help="Number of initial molecule conformations to sample from the database. \
              If equals to '-1' sample all conformations from the database.",
    )
    parser.add_argument(
        "--sample_initial_conformations",
        default=False,
        choices=[True, False],
        metavar="True|False",
        type=str2bool,
        help="Sample new conformation for every seed",
    )

    # Episode termination args
    parser.add_argument(
        "--timelimit_train", default=100, type=int, help="Max episode len on training"
    )
    parser.add_argument(
        "--timelimit_eval", default=100, type=int, help="Max episode len on evaluation"
    )
    parser.add_argument(
        "--terminate_on_negative_reward",
        default=True,
        choices=[True, False],
        metavar="True|False",
        type=str2bool,
        help="Terminate the episode when enough negative rewards are encountered",
    )
    parser.add_argument(
        "--max_num_negative_rewards",
        default=1,
        type=check_positive,
        help="Max number of negative rewards to terminate the episode",
    )

    # Reward args
    parser.add_argument(
        "--reward",
        choices=["rdkit", "dft"],
        default="rdkit",
        help="How the energy is calculated",
    )
    parser.add_argument(
        "--minimize_on_every_step",
        default=True,
        choices=[True, False],
        metavar="True|False",
        type=str2bool,
        help="Whether to minimize conformation with rdkit on every step",
    )

    # Backbone args
    parser.add_argument(
        "--backbone",
        choices=["schnet", "painn"],
        required=True,
        help="Type of backbone to use for actor and critic",
    )
    parser.add_argument(
        "--n_interactions",
        default=3,
        type=int,
        help="Number of interaction blocks for Schnet in actor/critic",
    )
    parser.add_argument(
        "--cutoff", default=5.0, type=float, help="Cutoff for Schnet in actor/critic"
    )
    parser.add_argument(
        "--n_rbf",
        default=50,
        type=int,
        help="Number of Gaussians for Schnet in actor/critic",
    )
    parser.add_argument(
        "--n_atom_basis",
        default=128,
        type=int,
        help="Number of features to describe atomic environments inside backbone",
    )

    # GOLF args
    parser.add_argument(
        "--actor",
        default="GOLF",
        type=str,
        choices=["GOLF", "rdkit"],
        help="Actor type. Rdkit can be used for evaluation only",
    )
    parser.add_argument(
        "--conformation_optimizer",
        default="LBFGS",
        type=str,
        choices=["GD", "Lion", "LBFGS", "Adam"],
        help="Conformation optimizer type",
    )
    parser.add_argument(
        "--conf_opt_lr",
        default=1.0,
        type=float,
        help="Initial learning rate for conformation optimizer.",
    )
    parser.add_argument(
        "--conf_opt_lr_scheduler",
        choices=["Constant", "CosineAnnealing"],
        default="Constant",
        help="Conformation optimizer learning rate scheduler type",
    )
    parser.add_argument(
        "--experience_saver",
        default="reward_threshold",
        choices=["reward_threshold", "last"],
        help="How to save experience to replay buffer",
    )
    parser.add_argument(
        "--store_only_initial_conformations",
        default=False,
        choices=[True, False],
        metavar="True|False",
        type=str2bool,
        help="For baseline experiments.",
    )

    # LBFGS args
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1,
        help="Number of iterations in the inner cycle LBFGS",
    )
    parser.add_argument(
        "--lbfgs_device",
        default="cuda",
        type=str,
        choices=["cuda", "cpu"],
        help="LBFGS device type",
    )

    # GD args
    parser.add_argument(
        "--momentum",
        default=0.0,
        type=float,
        help="Momentum argument for gradient descent confromation optimizer",
    )

    # Lion args
    parser.add_argument(
        "--lion_beta1",
        default=0.9,
        type=float,
        help="Beta_1 for Lion conformation optimizer",
    )
    parser.add_argument(
        "--lion_beta2",
        default=0.99,
        type=float,
        help="Beta_2 for Lion conformation optimizer",
    )

    # Training args
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size for both actor and critic",
    )
    parser.add_argument("--lr", default=3e-4, type=float, help="Actor learning rate")
    parser.add_argument(
        "--optimizer",
        default="adam",
        type=str,
        choices=["adam", "lion"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--lr_scheduler",
        default=None,
        type=none_or_str,
        choices=[None, "OneCycleLR", "CosineAnnealing", "StepLR"],
        help="LR scheduler",
    )
    parser.add_argument(
        "--clip_value", default=None, help="Clipping value for actor gradients"
    )
    parser.add_argument(
        "--energy_loss_coef",
        default=0.01,
        type=float,
        help="Weight for the energy part of the backbone loss",
    )
    parser.add_argument(
        "--force_loss_coef",
        default=1.0,
        type=float,
        help="Weight for the forces part of the backbone loss",
    )
    parser.add_argument(
        "--initial_conf_pct",
        default=0.0,
        type=float,
        help="Percentage of conformations from the initial database in each batch",
    )
    parser.add_argument(
        "--max_oracle_steps",
        default=1e6,
        type=int,
        help="Max number of oracle calls",
    )
    parser.add_argument(
        "--replay_buffer_size",
        default=1e5,
        type=int,
        help="Max capacity of the replay buffer",
    )
    parser.add_argument(
        "--utd_ratio",
        default=1,
        type=int,
        help="Number of NN updates per each data sample in replay buffer.\
              Total number of training steps = utd_ratio * max_oracle_steps",
    )
    parser.add_argument(
        "--subtract_atomization_energy",
        default=False,
        choices=[True, False],
        metavar="True|False",
        type=str2bool,
        help="Subtract atomization energy from the DFT energy for training",
    )
    parser.add_argument(
        "--action_norm_limit",
        default=0.05,
        type=float,
        help="Upper limit for action norm. Action norms larger get scaled down",
    )

    # Eval args
    parser.add_argument(
        "--eval_freq", default=1e3, type=int, help="Evaluation frequency"
    )
    parser.add_argument(
        "--n_eval_runs", default=10, type=int, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--eval_termination_mode",
        default="fixed_length",
        choices=["fixed_length", "grad_norm", "negative_reward"],
        help="When to terminate the episode on evaluation",
    )
    parser.add_argument(
        "--grad_threshold",
        default=1e-5,
        type=float,
        help="Terminates optimization when norm of the gradient is smaller than the threshold",
    )

    # Other args
    parser.add_argument(
        "--exp_name", required=True, type=str, help="Name of the experiment"
    )
    parser.add_argument("--seed", default=None, type=int, help="Random seed")
    parser.add_argument(
        "--full_checkpoint_freq",
        type=int,
        default=10000,
        help="How often full checkpoints are saved.\
              Note that only the most recent full checkpoint is available",
    )
    parser.add_argument(
        "--light_checkpoint_freq",
        type=int,
        default=10000,
        help="How often light checkpoints are saved",
    )
    parser.add_argument(
        "--save_checkpoints",
        default=False,
        choices=[True, False],
        metavar="True|False",
        type=str2bool,
        help="Save light and full checkpoints",
    )
    parser.add_argument(
        "--load_baseline",
        type=str,
        default=None,
        help="Checkpoint for the actor. Does not restore replay buffer",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Full checkpoint path (conformation optimizer and replay buffer)",
    )
    parser.add_argument("--log_dir", default=".", help="Directory where runs are saved")
    parser.add_argument(
        "--run_id", default="run-0", type=str, help="Run name in wandb project"
    )
    args = parser.parse_args()

    return args
