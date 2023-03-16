import argparse
import datetime
import json
import torch

from ase.db import connect
from pathlib import Path
from schnetpack.nn import BesselRBF

from env.make_envs import make_envs
from AL import DEVICE
from AL.AL_actor import Actor
from AL.eval import run_policy
from AL.utils import get_cutoff_by_string


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def run_agent(env, actor, n_confs, new_db):
    max_timestamps = env.unwrapped.TL
    if n_confs == -1:
        n_confs = env.unwrapped.get_db_length()

    for _ in range(n_confs):
        # Sample molecule from the test set
        env.reset()

        # Get initial energy
        if hasattr(env.unwrapped, "smiles"):
            smiles = env.unwrapped.smiles.copy()
        else:
            smiles = [None]
        fixed_atoms = env.unwrapped.atoms.copy()
        _ = run_policy(env, actor, fixed_atoms, smiles, max_timestamps)
        after_rl_atoms = env.unwrapped.atoms.copy()
        with connect(new_db) as conn:
            conn.write(after_rl_atoms[0])


def main(checkpoint_path, args, config):
    # Update config
    config.done_when_not_improved = False
    config.timelimit_eval = args.timelimit_eval
    config.n_parallel = 1
    config.reward = "rdkit"
    config.greedy = True
    config.sample_initial_conformations = False

    _, eval_env = make_envs(config)

    backbone_args = {
        "n_interactions": config.n_interactions,
        "n_atom_basis": config.n_atom_basis,
        "radial_basis": BesselRBF(n_rbf=config.n_rbf, cutoff=config.cutoff),
        "cutoff_fn": get_cutoff_by_string("cosine")(config.cutoff),
    }
    actor = Actor(
        backbone=config.backbone,
        backbone_args=backbone_args,
        action_scale=config.action_scale,
        action_norm_limit=config.action_norm_limit,
    )
    agent_path = checkpoint_path / args.agent_path
    actor.load_state_dict(torch.load(agent_path, map_location=torch.device(DEVICE)))
    actor.to(DEVICE)
    actor.eval()

    run_agent(eval_env, actor, args.conf_number, args.new_db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--agent_path", type=str, required=True)
    parser.add_argument(
        "--conf_number",
        default=int(1e5),
        type=int,
        help="Number of conformations to evaluate on",
    )
    parser.add_argument(
        "--timelimit_eval", default=1000, type=int, help="Length of evaluation episodes"
    )
    parser.add_argument("--db_path", default=str, required=True)
    parser.add_argument("--new_db_path", type=str, required=True)
    args = parser.parse_args()

    start_time = datetime.datetime.strftime(
        datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S"
    )
    checkpoint_path = Path(args.checkpoint_path)
    config_path = checkpoint_path / "config.json"
    # Read config and turn it into a class object with properties
    with open(config_path, "rb") as f:
        config = json.load(f)

    # TMP
    config["db_path"] = "/".join(config["db_path"].split("/")[-3:])
    config["eval_db_path"] = args.db_path
    config["molecules_xyz_prefix"] = "env/molecules_xyz"

    config = Config(**config)

    main(checkpoint_path, args, config)
