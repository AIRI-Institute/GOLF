# Gradual Optimization Learning for Conformational Energy Minimization
<p align="left">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This repository is the official implementation of Gradual Optimization Learning for Conformational Energy Minimization [[arxiv]](https://arxiv.org/abs/2311.06295).

<table border="1" class="dataframe">
   <thead>
      <tr style="text-align: center;">
         <th>Model</th>
         <th>$\overline{\text{pct}}_T (\%) \uparrow$</th>
         <th>$\text{pct}_{\text{div}} (\%) \downarrow$</th>
         <th>$\text{COV} (\%) \uparrow$ </th>
         <th>$\text{MAT} (\text{&#8491}) \downarrow $</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td><i>RDKit</i></td>
         <td><i>$84.92 \pm 10.6$</i></td>
         <td><i>$\mathbf{0.05}$</i></td>
         <td><i>$62.24$</i></td>
         <td><i>$0.509$</i></td>
      </tr>
      <tr>
         <td><i>Torsional Diffusion</i></td>
         <td><i>$25.63 \pm 21.4$</i></td>
         <td><i>$46.9$</i></td>
         <td><i>$11.3$</i></td>
         <td><i>$1.333$</i></td>
      </tr>
      <tr>
         <td><i>ConfOpt</i></td>
         <td><i>$36.48 \pm 23.0$</i></td>
         <td><i>$84.5$</i></td>
         <td><i>$19.88$</i></td>
         <td><i>$1.05$</i></td>
      </tr>
      <tr>
         <td><i>Uni-Mol+</i></td>
         <td><i>$62.20 \pm 17.2$</i></td>
         <td><i>$2.8$</i></td>
         <td><i>$68.79$</i></td>
         <td><i>$0.407$</i></td>
      </tr>
      <tr>
         <td><i>$f^{\text{baseline}}$</i></td>
         <td><i>$76.8 \pm 22.4$</i></td>
         <td><i>$7.5$</i></td>
         <td><i>$65.22$</i></td>
         <td><i>$0.482$</i></td>
      </tr>
      <tr>
         <td><i>$f^{\text{rdkit}}$</i></td>
         <td><i>$93.09 \pm 11.9$</i></td>
         <td><i>$3.8$</i></td>
         <td><i>$71.6$</i></td>
         <td><i>$0.426$</i></td>
      </tr>
      <tr>
         <td><i>$f^{\text{traj-10k}}$</i></td>
         <td><i>$95.3 \pm 7.3$ </i></td>
         <td><i>$4.5$</i></td>
         <td><i>$70.55$</i></td>
         <td><i>$0.440$</i></td>
      </tr>
      <tr>
         <td><i>$f^{\text{traj-100k}}$</i></td>
         <td><i>$96.3 \pm 9.8$</i></td>
         <td><i>$2.9$</i></td>
         <td><i>$71.43$</i></td>
         <td><i>$0.432$</i></td>
      </tr>
      <tr>
         <td><i>$f^{\text{traj-500k}}$</i></td>
         <td><i>$98.4 \pm 9.2$</i></td>
         <td><i>$1.8$</i></td>
         <td><i>$72.15$</i></td>
         <td><i>$0.442$</i></td>
      </tr>
      <tr>
         <td><i>$f^{\text{GOLF-1k}}$</i></td>
         <td><i>$98.5 \pm 5.3$</i></td>
         <td><i>$3.6$</i></td>
         <td><i></i>$76.54$</td>
         <td><i>$\mathbf{0.349}$</i></td>
      </tr>
      <tr>
         <td><i>$f^{\text{GOLF-10k}}$</i></td>
         <td><i>$\mathbf{99.4 \pm 5.2}$</i></td>
         <td><i>$2.4$</i></td>
         <td><i>$\mathbf{76.84}$</i></td>
         <td><i>$0.355$</i></td>
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
   wget https://shorturl.at/gknC5
   wget https://shorturl.at/jTXZ7
   cd ../
   ```
4. Train baseline PaiNN model
   ```
   cd scripts/train
   ./run_training_baseline.sh <cuda_device_number>
   ```
   Running this script will create a folder in the specified `log_dir` directory (we use "./results" in our configs and scripts). The name of the folder is specified by the `exp_name` hyperparameter. The folder will contain checkpoints, a metrics file and a config file with hyperparameters.

## Training the NNP on optimization trajectories
1. Set up environment on the GPU machine like in [the first section](#training-the-nnp-baseline)
2. Download optimization trajectories datasets.
   ```
   cd data
   wget https://shorturl.at/MV047
   wget https://shorturl.at/szJSZ
   wget https://shorturl.at/gnPTZ
   cd ../
   ```
3. Train PaiNN.
   ```
   cd scripts/train
   ./run_training_trajectories-10k.sh <cuda_device_number>
   ./run_training_trajectories-100k.sh <cuda_device_number>
   ./run_training_trajectories-500k.sh <cuda_device_number>
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
./run_training_GOLF-10k.sh <cuda_device_number>
```

## Evaluating NNPs
The evaluation can be done with or without `psi4` energy estimation for NNP-optimization trajectories. The argument 'eval_early_stop_steps' controls for which conformations in the optimization to evaluate energy/forces with `psi4`. For example, setting `eval_early_stop_steps` to an empty list will result in no additional `psi4` energy estimations, and setting it  to `[1 2 3 5 8 13 21 30 50 75 100]` will result in 13 additional energy evaluations for each conformation in evaluation dataset. Note that in order to compute the $\overline{pct}_T$, optimal energies obtained with the genuine oracle $\mathcal{O}$ (we used `psi4.optimize` for $\leq 200$ steps) must be available.

In this repo, we provide NNPs pre-trained on different datasets in the `checkpoints` directory:
   - $f^{\text{baseline}}$  (`checkpoints/baseline-NNP/NNP_checkpoint`)
   - $f^{\text{traj-10k}}$ (`checkpoints/traj-10k/NNP_checkpoint`)
   - $f^{\text{traj-100k}}$ (`checkpoints/traj-100k/NNP_checkpoint`)
   - $f^{\text{traj-500k}}$ (`checkpoints/trak-500k/NNP_checkpoint`)
   - $f^{\text{GOLF-1k}}$ (`checkpoints/GOLF-1k/NNP_checkpoint`)
   - $f^{\text{GOLF-10k}}$ (`checkpoints/GOLF-10k/NNP_checkpoint`)

For example, to evaluate GOLF-10k and additionally calculate `psi4` energies/forces along the optimization trajectory, run:
```
python evaluate_batch_dft.py --checkpoint_path checkpoints/GOLF-10k --agent_path NNP_checkpoint_actor --n_parallel 240 --n_threads 24 --conf_number -1 --host_file_path env/host_names.txt --eval_db_path data/GOLF_test.db --timelimit 100 --terminate_on_negative_reward False --reward dft --minimize_on_every_step False --eval_early_stop_steps 1 2 3 5 8 13 21 30 50 75 100
```
Make sure that `n_threads` is equal to the number of workers on each CPU-rich machine. Setting `n_threads` to a larger number will result in optimization failures. If you wish to only evaluate the last state in each optimization trajectory, set `timelimit` and `eval_early_stop_steps` to the same number: `--timelimit T --eval_early_stop_steps T`.

After the evaluation is finished, an `evaluation_metrics.json` file with per-step metrics is created. Each record in `evaluation_metrics.json` describes optimization statistics for a single conformation and contains such metrics as: forces/energies MSE, percentage of optimized energy, predicted and ground-truth energies, etc. The final NNP-optimized conformations are stored in `results.db` database.
