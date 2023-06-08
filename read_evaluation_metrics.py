import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import wandb


def mean_std(values, min_threshold=-math.inf, max_threshold=math.inf):
    values = np.asarray(values)
    mask = (min_threshold <= values) & (values <= max_threshold)
    values = values[mask]

    return values.mean(), values.std()


def outliers_fraction(values, min_threshold=-math.inf, max_threshold=math.inf):
    values = np.asarray(values)
    min_mask = values < min_threshold
    min_mean = values[min_mask].mean()
    max_mask = max_threshold < values
    max_mean = values[max_mask].mean()

    return (min_mask.mean(), min_mean), (max_mask.mean(), max_mean)


def read_metrics(path):
    metrics = {}
    with open(path, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == '':
                continue

            record = json.loads(line)
            metrics[record['conformation_id']] = record

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    metrics = read_metrics(args.path)
    pct_prefix = 'pct_of_minimized_energy@step:'
    energy_rdkit_prefix = 'rdkit_energy@step:'
    n_iter_prefix = 'n_iter@step:'
    energy_mse_prefix = 'energy_mse@step:'
    force_mse_prefix = 'force_mse@step:'
    not_finite_action_step = 'not_finite_action_step'
    rdkit_initial_energy = 'rdkit_initial_energy'
    rdkit_final_energy = 'rdkit_final_energy'

    stats = []
    for conformation_id, record in metrics.items():
        pct = {}
        n_iter = {}
        energy_rdkit = {}
        energy_mse = {}
        force_mse = {}
        invalid_step = record[not_finite_action_step]
        for key, value in record.items():
            if key.startswith(pct_prefix):
                step = int(key[len(pct_prefix):])
                pct[step] = value
            elif key.startswith(energy_mse_prefix):
                step = int(key[len(energy_mse_prefix):])
                energy_mse[step] = value
            elif key.startswith(force_mse_prefix):
                step = int(key[len(force_mse_prefix):])
                force_mse[step] = value
            elif key.startswith(n_iter_prefix):
                step = int(key[len(n_iter_prefix):])
                n_iter[step] = value
            elif key.startswith(energy_rdkit_prefix):
                step = int(key[len(energy_rdkit_prefix):])
                energy_rdkit[step] = value

        for step in sorted(pct.keys()):
            if 0 <= invalid_step <= step:
                continue
            stats.append(
                {'conformation_id': conformation_id, 'step': step, 'n_iter': n_iter[step],
                 'energy_rdkit': energy_rdkit[step], 'pct': pct[step], 'energy_mse': energy_mse[step],
                 'force_mse': force_mse[step]}
            )

    print(f'{"conformation_id":15} {"step":6} {"n_iter":6} {"energy_rdkit":20} {"pct":20} {"energy_mse":20} {"force_mse":20}')
    for record in stats:
        conformation_id = record['conformation_id']
        step = record['step']
        n_iter = record['n_iter']
        pct = record['pct']
        energy_mse = record['energy_mse']
        force_mse = record['force_mse']
        energy_rdkit = record['energy_rdkit']
        print(f'{conformation_id:<15} {step:<6} {n_iter:<6} {energy_rdkit:<20} {pct:<20} {energy_mse:<20} {force_mse:<20}')
