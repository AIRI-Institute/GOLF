import argparse


def str2bool(s):
    """ helper function used in order to support boolean command line arguments """
    if s.lower() in ('true', 't', '1'):
        return True
    elif s.lower() in ('false', 'f', '0'):
        return False
    else:
        return s

def none_or_str(value):
    if value == 'None':
        return None
    return value

def get_args():
    parser = argparse.ArgumentParser()

    # Env args
    parser.add_argument(
        "--n_parallel",
        default=1,
        type=int,
        help="Number of copies of env to run in parallel")
    parser.add_argument(
        "--n_threads",
        default=1,
        type=int,
        help="Number of parallel threads for DFT computations")
    parser.add_argument(
        "--db_path",
        default="env/data/malonaldehyde.db",
        type=str,
        help="Path to molecules database for training")
    parser.add_argument(
        "--eval_db_path",
        default="",
        type=str,
        help="Path to molecules database for evaluation")
    parser.add_argument(
        "--num_initial_conformations",
        default=50000,
        type=int,
        help="Number of initial molecule conformations to sample from the database. \
              If equals to '-1' sample all conformations from the database.")
    parser.add_argument(
        "--sample_initial_conformations",
        default=False,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Sample new conformation for every seed")

    # Timelimit args
    parser.add_argument(
        "--timelimit",
        default=100,
        type=int,
        help="Timelimit for MD env")
    parser.add_argument(
        "--terminate_on_negative_reward",
        default=True,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Terminate the episode when enough negative rewards are encountered")
    parser.add_argument(
        "--max_num_negative_rewards",
        default=1,
        type=int,
        help="Max number of negative rewards to terminate the episode")

    # Reward args
    parser.add_argument(
        "--reward",
        choices=["rdkit", "dft"],
        default="rdkit",
        help="How the energy is calculated")
    parser.add_argument(
        "--minimize_on_every_step",
        default=True,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Whether to minimize conformation with rdkit on every step")
    parser.add_argument(
        "--M",
        type=int,
        default=10,
        help="Number of steps to run rdkit minimization for")
    parser.add_argument(
        "--molecules_xyz_prefix",
        type=str,
        default="",
        help="Path to env/ folder. For cluster compatability")

    # Backbone args
    parser.add_argument(
        "--backbone",
        choices=["schnet", "painn"],
        required=True,
        help="Type of backbone to use for actor and critic")
    parser.add_argument(
        "--n_interactions",
        default=3,
        type=int,
        help="Number of interaction blocks for Schnet in actor/critic")
    parser.add_argument(
        "--cutoff",
        default=5.0,
        type=float,
        help="Cutoff for Schnet in actor/critic")
    parser.add_argument(
        "--n_rbf",
        default=50,
        type=int,
        help="Number of Gaussians for Schnet in actor/critic")
    parser.add_argument(
        '--use_cosine_between_vectors',
        default=True,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Use cosine of vectors instead of scalar product in PaiNN")

    # AL args
    parser.add_argument(
        "--action_scale",
        default=0.01,
        type=float,
        help="Multiply actions by action_scale.")
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size for both actor and critic")
    parser.add_argument(
        "--lr",
        default=3e-4,
        type=float,
        help="Actor learning rate")
    parser.add_argument(
        "--clip_value",
        default=None,
        help="Clipping value for actor gradients")
    parser.add_argument(
        "--lr_scheduler",
        default=None,
        type=none_or_str,
        choices=[None, "OneCycleLR", "StepLR"],
        help="LR scheduler")
    parser.add_argument(
        "--action_norm_limit",
        default=0.05,
        type=float,
        help="Limit max action norm")
    parser.add_argument(
        "--energy_loss_coef",
        default=0.01,
        type=float)
    parser.add_argument(
        "--force_loss_coef",
        default=1.,
        type=float)
    parser.add_argument(
        "--group_by_n_atoms",
        default=False,
        choices=[True, False],
        metavar='True|False', type=str2bool,
        help="Partition batch into groups by molecule size.\
              For PaiNN + schnetpack=1.0.0")
    parser.add_argument(
        "--store_only_initial_conformations",
        default=False,
        choices=[True, False],
        metavar='True|False', type=str2bool,
        help="For baseline experiments.")
    
    # Eval args
    parser.add_argument(
        "--eval_freq",
        default=1e3,
        type=int,
        help="Evaluation frequency")
    parser.add_argument(
        "--n_eval_runs",
        default=10,
        type=int,
        help="Number of evaluation episodes")
    parser.add_argument(
        "--n_explore_runs",
        default=5,
        type=int,
        help="Number of exploration episodes during evaluation")
    parser.add_argument(
        "--evaluate_multiple_timesteps",
        default=False,
        choices=[True, False],
        metavar='True|False', type=str2bool,
        help="Evaluate at multiple timesteps")

    # Other args
    parser.add_argument(
        "--exp_name",
        required=True,
        type=str,
        help="Name of the experiment")
    parser.add_argument(
        "--replay_buffer_size",
        default=int(1e5),
        type=int,
        help="Size of replay buffer")
    parser.add_argument(
        "--max_timesteps",
        default=1e6,
        type=int,
        help="Max time steps to run environment")
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Random seed")
    parser.add_argument(
        "--full_checkpoint_freq",
        type=int,
        default=10000,
        help="How often full checkpoints are saved.\
              Note that only the most recent full checkpoint is available")
    parser.add_argument(
        "--light_checkpoint_freq",
        type=int,
        default=10000,
        help="How often light checkpoints are saved")
    parser.add_argument(
        "--save_checkpoints",
        default=False,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Save light and full checkpoints")
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Path to load the model from")
    parser.add_argument(
        "--log_dir",
        default='.',
        help="Directory where runs are saved")
    parser.add_argument(
        "--run_id",
        default='run-0',
        type=str,
        help="Run name in wandb project")
    args = parser.parse_args()

    return args