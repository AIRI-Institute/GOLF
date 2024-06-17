# <img src="https://latex.codecogs.com/svg.image?\huge\mathbf{\nabla^2}\textbf{DFT}" title="\Large \mathbf{\nabla^2\text{DFT}}" /> geometry optimization evaluation manual

This repository provides an example of how to evaluate the _SchNet_ and _PaiNN_ models in the geometry optimization task. All the information about the <img src="https://latex.codecogs.com/svg.image?\mathbf{\nabla^2}\textbf{DFT}" title="\mathbf{\nabla^2\text{DFT}}" /> dataset and benchmark can be found in [nablaDFT](https://github.com/AIRI-Institute/nablaDFT) repo.


## Evaluate _SchNet_ or _PaiNN_
All the checkpoints with the corresponding configs are in `checkpoints` directory.
1. **Set up virtual environment on the parent machine.** To perform optimization in a reasonable time, we assume that the parent machine has a GPU.
   ```
   source setup_gpu_env.sh
   ```
2. **Set up dft workers on CPU-rich machines.** To speed up the evaluation, we parallelize DFT computations using several CPU-rich machines. Hostnames of CPU-rich machines must be provided in `env/host_names.env`. If CPU-rich machines are unavailable, the Psi4 computations can be performed on the parent machine. In this case, `env/host_names.env` must contain ip-address of the parent machine.
   - Log in to CPU-rich machines. They must be accessible via `socket.bind`.
   - Clone GOLF repo `git clone --branch nabla2DFT-eval https://github.com/AIRI-Institute/GOLF`.
   - Set up virtual environment for DFT workers
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
3. **Download evaluation dataset $\mathcal{D}_{\text{opt}}^{\text{test}}$ on the parent machine**:
  ```
  mkdir data && cd data
  wget https://a002dlils-kadurin-nabladft.obs.ru-moscow-1.hc.sbercloud.ru/data/nablaDFTv2/energy_databases/test_trajectories_initial.db
  cd ../
  ```
4. **Run evaluation on the parent machine**:
   ```
   CUDA_VISIBLE_DEVICES=0 python evaluate_batch_dft.py --checkpoint_path checkpoints/painn-100k --agent_path painn_100k.ckpt --n_parallel 60 --n_workers 8 --conf_number -1 --host_file_path env/host_names.txt --eval_db_path data/test_trajectories_initial.db --timelimit 100 --terminate_on_negative_reward False --reward dft --minimize_on_every_step False --eval_early_stop_steps 100 --subtract_atomization_energy False
   ```
   - `checkpoint_path` path to directory that contains model checkpoint and a config. In this repo, all model checkpoints are in `checkpoints` directory. You can find checkpoints for other models [here](https://github.com/AIRI-Institute/nablaDFT/blob/main/nablaDFT/links/models_checkpoints.json). The configs with models' hyperparameters can be found [here](https://github.com/AIRI-Institute/nablaDFT/tree/main/config/model).
   - `eval_db_path` path to the database of molecular conformations (in SQLite3 format). 
   - `n_parallel` batch size for NNP.
   - `n_workers` number of DFT workers. Must match the number of workers on each CPU-rich machine.
   - 'eval_early_stop_steps' controls for which conformations in the optimization trajectory to evaluate energy/forces with `psi4`. For example, setting `eval_early_stop_steps` to an empty list will result in no additional `psi4` energy estimations, setting it  to `[100]` will result in one estimation, etc. If you wish to only evaluate the last state in each optimization trajectory, set `timelimit` and `eval_early_stop_steps` to the same number: `--timelimit T --eval_early_stop_steps T`.
  
After the evaluation is finished, an `evaluation_metrics.json` file with per-step metrics will be created. Each record in `evaluation_metrics.json` describes optimization statistics for a single conformation and contains such metrics as: forces/energies MSE, percentage of optimized energy, predicted and ground-truth energies, etc. The final NNP-optimized conformations are stored in `results.db` database.
