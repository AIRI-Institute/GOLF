import argparse
import numpy as np
import pickle
import plotly.graph_objects as go

from env.xyz2mol import parse_molecule


def plot_trajectories(molecule, trajectory, N, save_figure_path):
    # Legend groups
    legend_groups_RL = []
    for a in molecule.GetAtoms():
        legend_groups_RL.append(a.GetSymbol() + '_RL')
    legend_groups = []
    for a in molecule.GetAtoms():
        legend_groups.append(a.GetSymbol())
    # Showlegend
    traces_RL = [False if (len(legend_groups_RL[:i])>0 and l in legend_groups_RL[:i]) 
            else True for i, l in enumerate(legend_groups_RL)]
    traces = [False if (len(legend_groups[:i])>0 and l in legend_groups[:i]) 
            else True for i, l in enumerate(legend_groups)]
    # Colors
    cm_RL = {'C_RL': 'black', 'O_RL': 'purple', 'H_RL': 'orange'}
    cm = {'C': 'grey', 'O': 'red', 'H': 'yellow'}

    fig = go.Figure()
    # Plot RL trajectories
    for i in range(molecule.GetNumAtoms()):
        fig.add_trace(
            go.Scatter3d(
                x=trajectory[:N + 1, i, 0],
                y=trajectory[:N + 1, i, 1],
                z=trajectory[:N + 1, i, 2],
                mode='lines',
                marker=dict(
                    size=2,
                    color=cm_RL[legend_groups_RL[i]],
                    colorscale='Viridis',
                ),
                line={"color":cm_RL[legend_groups_RL[i]], },
                legendgroup=legend_groups_RL[i],
                showlegend=traces_RL[i],
                name="{}".format(legend_groups_RL[i])
            )
        )
    # Plot rdkit trajectories
    for i in range(molecule.GetNumAtoms()):
        fig.add_trace(
            go.Scatter3d(
                x=trajectory[N:, i, 0],
                y=trajectory[N:, i, 1],
                z=trajectory[N:, i, 2],
                mode='lines',
                marker=dict(
                    size=2,
                    color=cm[legend_groups[i]],
                    colorscale='Viridis',
                ),
                line={"color":cm[legend_groups[i]], },
                legendgroup=legend_groups[i],
                #hovertext=labels[i],
                showlegend=traces[i],
                name="{}".format(legend_groups[i])
            )
        )
    # Display figure
    fig.update_traces(mode='markers+lines')
    fig.show()
    if save_figure_path is not None:
        fig.write_html(save_figure_path)

def main(args):
    molecule = parse_molecule(args.molecule_path)
    with open(args.trajectory_path, 'rb') as handle:
        trajectories = pickle.load(handle)
    plot_trajectories(molecule, trajectories, args.N, args.save_figure_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule_path", default="env/molecules_xyz/malonaldehyde.xyz", type=str, help="Path to example .xyz file")
    parser.add_argument("--trajectory_path", required=True, type=str, help="File with pickled trajectory")
    parser.add_argument("--N", default=5, type=int, help="Run RL policy for N steps")
    parser.add_argument("--save_figure_path", type=str, default=None, help="Save plot. Must be an html file")
    args = parser.parse_args()

    main(args)