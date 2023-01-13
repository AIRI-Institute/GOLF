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
    # Algorthm
    parser.add_argument(
        "--algorithm",
        default='TQC',
        choices=['TQC', 'PPO', 'SAC', 'GD'])

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
        "--sample_initial_conformation",
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
        "--increment_timelimit",
        default=False,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Whether to increment timelimit during training")
    parser.add_argument(
        "--timelimit_step",
        default=10,
        type=int,
        help="By which number to increment timelimit")
    parser.add_argument(
        "--timelimit_interval",
        default=150000,
        type=int,
        help="How often to increment timelimit")
    parser.add_argument(
        "--greedy",
        default=False,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Returns done on every step independent of the timelimit")
    parser.add_argument(
        "--done_on_timelimit",
        default=False,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Env returns done when timelimit is reached")
    parser.add_argument(
        "--done_when_not_improved",
        default=True,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Return done if energy has not improved")

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

    # Action scale args.
    parser.add_argument(
        "--action_scale",
        default=0.01,
        type=float,
        help="Multiply actions by action_scale.")
    parser.add_argument(
        "--target_entropy_action_scale",
        default=0.01,
        type=float,
        help="Controls target entropy of the distribution")

    # Backbone args
    parser.add_argument(
        "--backbone",
        choices=["schnet", "painn"],
        required=True,
        help="Type of backbone to use for actor and critic"
    )
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

    # Policy args
    parser.add_argument(
        "--out_embedding_size",
        default=128,
        type=int,
        help="Output embedding size for policy")
    parser.add_argument(
        "--limit_actions",
        default=False,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Whether to limit action norms with tanh")
    parser.add_argument(
        "--generate_action_type",
        choices=["delta_x", "spring_and_mass"],
        required=True,
        help="Type of action generation block to use")
    parser.add_argument(
        "--cutoff_type",
        choices=["hard", "cosine"],
        required=True,
        help="Type of cutoff to use in action generation block")
    parser.add_argument(
        "--use_activation",
        default=False,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="If True additionally process atom embeddings before generating action")
    parser.add_argument(
        "--summation_order",
        default="to",
        choices=["to", "from"],
        help="If 'to' then action is calculated by summing all vectors coming to atom.\
              If 'from' then action is calculated by summing all vectors coming from atom")
    parser.add_argument(
        "--n_quantiles",
        default=25,
        type=int,
        help="Number of quantiles in each net in critic")
    parser.add_argument(
        "--n_nets",
        default=5,
        type=int,
        help="Total number of nets in critic")
    parser.add_argument(
        '--m_nets',
        default=2,
        type=int,
        help="Number of nets randomly sampled for update on each step"
    )

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

    # TQC args
    parser.add_argument(
        "--pretrain_critic",
        default=0,
        type=int,
        help="Number of steps to pretrain critic on random actions")
    parser.add_argument(
        "--top_quantiles_to_drop_per_net",
        default=2,
        type=int,
        help="Number of quantiles to drop per net in target")
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size for both actor and critic")
    parser.add_argument(
        "--discount",
        default=0.99,
        type=float,
        help="Discount factor")
    parser.add_argument(
        "--tau",
        default=0.005,
        type=float,
        help="Target network update rate")
    parser.add_argument(
        "--actor_lr",
        default=3e-4,
        type=float,
        help="Actor learning rate")
    parser.add_argument(
        "--actor_clip_value",
        default=None,
        help="Clipping value for actor gradients")
    parser.add_argument(
        "--critic_lr",
        default=3e-4,
        type=float,
        help="Critic learning rate")
    parser.add_argument(
        "--critic_clip_value",
        default=None,
        help="Clipping value for critic gradients")
    parser.add_argument(
        "--lr_scheduler",
        default=None,
        type=none_or_str,
        choices=[None, "OneCycleLR", "StepLR"],
        help="LR scheduler")
    parser.add_argument(
        "--alpha_lr",
        default=3e-4,
        type=float,
        help="Alpha learning rate")
    parser.add_argument(
        "--initial_alpha",
        default=1.0,
        type=float,
        help="Initial value for alpha")

    # PPO args
    parser.add_argument(
        "--clip_param",
        default=0.2,
        type=float,
        help="PPO clip value")
    parser.add_argument(
        "--ppo_epoch",
        default=4,
        type=int,
        help="Number of epochs for PPO update")
    parser.add_argument(
        "--num_mini_batch",
        default=16,
        type=int,
        help="Number of minibatches per ppo_epoch")
    parser.add_argument(
        "--value_loss_coef",
        default=0.5,
        type=float, 
        help="Weight for value loss")
    parser.add_argument(
        "--entropy_coef",
        default=0.01,
        type=float,
        help="Entropy coefficient")
    parser.add_argument(
        "--use_clipped_value_loss",
        default=True,
        type=bool,
        help="Whether to use clipped value loss")

    # GD args
    parser.add_argument(
        "--action_norm_limit",
        default=0.05,
        type=float,
        help="Limit max action norm")
    
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
        "--update_frequency",
        default=1,
        type=int,
        help="How often agent is updated")
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