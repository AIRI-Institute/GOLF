# <img src="https://latex.codecogs.com/svg.image?\huge\mathbf{\nabla^2}\textbf{DFT}" title="\Large \mathbf{\nabla^2\text{DFT}}" /> geometry optimization evaluation manual

This repository provides an example of how to evaluate the _SchNet_ and _PaiNN_ models in the geometry optimization task. All the information about the <img src="https://latex.codecogs.com/svg.image?\mathbf{\nabla^2}\textbf{DFT}" title="\mathbf{\nabla^2\text{DFT}}" /> dataset and benchmark can be found in [nablaDFT](https://github.com/AIRI-Institute/nablaDFT) repo. Links to all checkpoints are in [model checkpoints](https://github.com/AIRI-Institute/nablaDFT/blob/main/nablaDFT/links/models_checkpoints.json).


## Evaluate _SchNet_ or _PaiNN_
All the checkpoints with the corresponding configs are in `checkpoints` directory.
1. **Setup virtual environment on the parent machine.** To perform optimization in a reasonable time, we assume that the parent machine has a GPU.
   ```
   source setup_gpu_env.sh
   ```
2. **Setup dft workers on CPU-rich machines.** To speed up the evaluation, we parallelize DFT computations using several CPU-rich machines. Hostnames of CPU-rich machines must be provided in the `env/host_names.env`. If CPU-rich machines are unavailable, the Psi4 computations can be performed on the parent machine. In this case, `env/host_names.env` must contain ip-address of the parent machine.
   - Log in to CPU-rich machines. They must be accessible via `socket.bind`.
   - Clone GOLF repo `git clone --branch nabla2DFT-eval https://github.com/AIRI-Institute/GOLF`.
   - Set up verual environments for DFT workers
  ```
  cd scripts && source setup_host.sh <number of threads per worker> <number of workers> <starting port number>
  ```
  For example, to launch 4 workers using 4 threads each and listening to ports `20000, 20001, 20002, 20003`, run:
  ```
  cd scripts && source setup_host.sh 4 4 20000
  ```
  ```
  cd checkpoints/painn-100k
  wget https://a002dlils-kadurin-nabladft.obs.ru-moscow-1.hc.sbercloud.ru/data/nablaDFTv2/models_checkpoints/PaiNN/painn_100k.ckpt && mv painn_100k.ckpt NNP_checkpoint_actor
  ```
2. **Download evaluation dataset $\mathcal{D}_{\text{opt}}^{\text{test}}$ on the parent machine**:
  ```
  wget https://a002dlils-kadurin-nabladft.obs.ru-moscow-1.hc.sbercloud.ru/data/nablaDFTv2/energy_databases/test_trajectories_initial.db
  ```
4. To speed up the evaluation, we parallelize DFT computations using several CPU-rich machines. Hostnames of CPU-rich machines must be provided in the `env/host_names.env`. If CPU-rich machines are unavailable, the Psi4 computations can be performed on the GPU machine. In this case, `env/host_names.env` must be changed accordingly.
   Set up workers that perform Psi4 calculations in parallel:
   - Log in to CPU-rich machines. They must be accessible via `socket.bind`.
   - Set up environments on CPU-rich machines.
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
The evaluation can be done with or without `psi4` energy estimation for NNP-optimization trajectories. The argument 'eval_early_stop_steps' controls for which conformations in the optimization trajectory to evaluate energy/forces with `psi4`. For example, setting `eval_early_stop_steps` to an empty list will result in no additional `psi4` energy estimations, and setting it  to `[1 2 3 5 8 13 21 30 50 75 100]` will result in 13 additional energy evaluations for each conformation in evaluation dataset. Note that in order to compute the $\overline{pct}_T$, optimal energies obtained with the genuine oracle $\mathcal{O}$ must be available. In our work, `psi4.optimize` with spherical representation of the molecule was used (approximately 30 steps until convergence).

In this repo, we provide NNPs pre-trained on different datasets and with GOLF in the `checkpoints` directory:
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

After the evaluation is finished, an `evaluation_metrics.json` file with per-step metrics will be created. Each record in `evaluation_metrics.json` describes optimization statistics for a single conformation and contains such metrics as: forces/energies MSE, percentage of optimized energy, predicted and ground-truth energies, etc. The final NNP-optimized conformations are stored in `results.db` database.

