# Gradual Optimization Learning for Conformational Energy Minimization
<p align="left">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This repository is the official implementation of Gradual Optimization Learning for Conformational Energy Minimization [[openreview]](https://openreview.net/forum?id=FMMF1a9ifL).

**Experiments and results on the [nablaDFT](https://pubs.rsc.org/en/content/articlelanding/2022/cp/d2cp03966d) dataset can be found in the "main" branch.**

<table border="1" class="dataframe">
   <thead>
      <tr style="text-align: center;">
         <th>Model</th>
         <th>$\overline{\text{pct}}_T (\%) \uparrow$</th>
         <th>$\text{pct}_{\text{div}} (\%) \downarrow$</th>
         <th>$\overline{E^{\text{res}}}_T\tiny{\text{(kc/mol)}}\downarrow$</th>
         <th>$\text{pct}_{\text{success}} (\%) \uparrow$</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td><i>$f^{\text{baseline}}$</i></td>
         <td><i>$90.4 \pm 12.0$</i></td>
         <td><i>$4.7$</i></td>
         <td><i>$3.6$</i></td>
         <td><i>$19.7$</i></td>
      </tr>
      <tr>
         <td><i>$f^{\text{traj-10k}}$</i></td>
         <td><i>$93.4 \pm 10.0$ </i></td>
         <td><i>$6.8$</i></td>
         <td><i>$2.4$</i></td>
         <td><i>$37.4$</i></td>
      </tr>
      <tr>
         <td><i>$f^{\text{traj-100k}}$</i></td>
         <td><i>$\mathbf{94.3 \pm 9.4}$</i></td>
         <td><i>$\mathbf{2.4}$</i></td>
         <td><i>$\mathbf{2.1}$</i></td>
         <td><i>$\mathbf{44.2}$</i></td>
      </tr>
      <tr>
         <td><i>$f^{\text{traj-220k}}$</i></td>
         <td><i>$93.9 \pm 9.6$</i></td>
         <td><i>$\mathbf{2.4}$</i></td>
         <td><i>$2.3$</i></td>
         <td><i>$41.6$</i></td>
      </tr>
      <tr>
         <td><i>$f^{\text{GOLF-10k}}$</i></td>
         <td><i>$\mathbf{94.2 \pm 8.9}$</i></td>
         <td><i>$3.2$</i></td>
         <td><i>$\mathbf{2.1}$</i></td>
         <td><i>$40.9$</i></td>
      </tr>
   </tbody>
</table>

## Training the NNP baseline
1. Set up environment on the GPU machine.
   ```
   # On the GPU machine
   conda create -y -n GOLF_env python=3.9
   conda activate GOLF_env
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install  psi4 -c psi4
   python -m pip install -r requirements.txt
   ```
2. Download training dataset $\mathcal{D}_0$ and evaluation dataset $\mathcal{D}\_{\text{test}}$
   ```
   mkdir data && cd data
   wget https://sc.link/6Ljw1
   wget https://sc.link/HtZte
   cd ../
   ```
4. Train baseline PaiNN model
   ```
   cd scripts/train
   ./run_training_baseline_spice.sh <cuda_device_number>
   ```
   Running this script will create a folder in the specified `log_dir` directory (we use "./results" in our configs and scripts). The name of the folder is specified by the `exp_name` hyperparameter. The folder will contain checkpoints, a metrics file and a config file with hyperparameters.

## Training the NNP on optimization trajectories
1. Set up environment on the GPU machine like in [the first section](#training-the-nnp-baseline)
2. Download optimization trajectories datasets.
   ```
   cd data
   wget https://sc.link/7tgCE
   wget https://sc.link/AjsfR
   wget https://sc.link/yd08k
   cd ../
   ```
3. Train PaiNN.
   ```
   cd scripts/train
   ./run_training_trajectories_10k_spice.sh <cuda_device_number>
   ./run_training_trajectories_100k_spice.sh <cuda_device_number>
   ./run_training_trajectories_500k_spice.sh <cuda_device_number>
   ```

## Training NNPs with GOLF

### Distributed Gradient Calculation with Psi4
To speed up the training, we parallelize DFT computations using several CPU-rich machines. The training of the NNP takes place on the parent machine with a GPU.
1. Set up environment on the GPU machine like in [the first section](#training-the-nnp-baseline)
1. Log in to CPU-rich machines. They must be accessible via `ssh`.
2. Set up environments on CPU-rich machines.
   ```
   # On CPU rich machines
   git clone https://github.com/AIRI-Institute/GOLF
   cd GOLF/scripts
   ./setup_host.sh n_ports ports_range_begin
   ```
   Here, `n_ports` denotes number of workers on a CPU-rich machine, and `ports_range_begin` denotes the starting port numbers for workers. Workers calculate energies and forces using `psi4` for newly generated conformations. For example, `./setup_host.sh 24 20000` will launch a total of 48 workers listening to ports `20000, ... , 20023`. You can change the `ports_range_begin` in `env/dft.py`.
   
   By default we assume that each worker uses 4 CPU-cores (can be changed in `env/dft_worker.py`, line 22) which means that `n_ports` must be less or equal to `total_cpu_number / 4`.
4. Add ip addresses of CPU rich machines to a text file. We use `env/host_names.txt`.

### Training with GOLF
Train PaiNN with GOLF.
```
cd scripts/train
./run_training_GOLF_10k_SPICE.sh <cuda_device_number>
```

## Evaluating NNPs
The evaluation can be done with or without `psi4` energy estimation for NNP-optimization trajectories. The argument 'eval_early_stop_steps' controls for which conformations in the optimization trajectory to evaluate energy/forces with `psi4`. For example, setting `eval_early_stop_steps` to an empty list will result in no additional `psi4` energy estimations, and setting it  to `[1 2 3 5 8 13 21 30 50 75 100]` will result in 13 additional energy evaluations for each conformation in evaluation dataset. Note that in order to compute the $\overline{pct}_T$, optimal energies obtained with the genuine oracle $\mathcal{O}$ must be available. In our work, `psi4.optimize` with spherical representation of the molecule was used (approximately 30 steps until convergence).

In this repo, we provide NNPs pre-trained on different datasets in the `checkpoints` directory:
   - $f^{\text{baseline}}$  (`checkpoints/baseline-NNP-SPICE/NNP_checkpoint`)
   - $f^{\text{traj-10k}}$ (`checkpoints/traj-10k-SPICE/NNP_checkpoint`)
   - $f^{\text{traj-100k}}$ (`checkpoints/traj-100k-SPICE/NNP_checkpoint`)
   - $f^{\text{traj-500k}}$ (`checkpoints/trak-500k-SPICE/NNP_checkpoint`)
   - $f^{\text{GOLF-10k}}$ (`checkpoints/GOLF-10k-SPICE/NNP_checkpoint`)

For example, to evaluate GOLF-10k and additionally calculate `psi4` energies/forces along the optimization trajectory, run:
```
python evaluate_batch_dft.py --checkpoint_path checkpoints/GOLF-10k --agent_path NNP_checkpoint_actor --n_parallel 240 --n_threads 24 --conf_number -1 --host_file_path env/host_names.txt --eval_db_path data/GOLF_test.db --timelimit 100 --terminate_on_negative_reward False --reward dft --minimize_on_every_step False --eval_early_stop_steps 1 2 3 5 8 13 21 30 50 75 100
```
Make sure that `n_threads` is equal to the number of workers on each CPU-rich machine. Setting `n_threads` to a larger number will result in optimization failures. If you wish to only evaluate the last state in each optimization trajectory, set `timelimit` and `eval_early_stop_steps` to the same number: `--timelimit T --eval_early_stop_steps T`.

After the evaluation is finished, an `evaluation_metrics.json` file with per-step metrics is created. Each record in `evaluation_metrics.json` describes optimization statistics for a single conformation and contains such metrics as: forces/energies MSE, percentage of optimized energy, predicted and ground-truth energies, etc. The final NNP-optimized conformations are stored in `results.db` database.
