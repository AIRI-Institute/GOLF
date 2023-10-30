# Gradual Optimization Learning for Conformational Energy Minimization
<p align="left">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

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
2. Download training dataset $\mathcal{D}_0$.
   ```
   wget https://n-usr-31b1j.s3pd12.sbercloud.ru/b-usr-31b1j-qz9/data/energy_dbs/GOLF_train.db
   ```
3. Train baseline PaiNN model
   ```
   cd scripts/train
   ./run_training_baseline.sh
   ```
   Running this script will create a folder in the specified `log_dir` directory. The name of the folder is specified by the `exp_name` hyperparameter. The folder will contain checkpoints, a metrics file and a config file with hyperparameters.

## Training the NNP on optimization trajectories
1. Set up environment on the GPU machine like in [the first section](#training-the-nnp-baseline)
2. Download optimization trajectories datasets.
   ```
   wget https://n-usr-31b1j.s3pd12.sbercloud.ru/b-usr-31b1j-qz9/data/energy_dbs/traj-10k.db
   wget https://n-usr-31b1j.s3pd12.sbercloud.ru/b-usr-31b1j-qz9/data/energy_dbs/traj-100k.db
   wget https://n-usr-31b1j.s3pd12.sbercloud.ru/b-usr-31b1j-qz9/data/energy_dbs/traj-500k.db
   ```
3. Train PaiNN.
   ```
   cd scripts/train
   ./run_training_trajectories-10k.sh
   ./run_training_trajectories-100k.sh
   ./run_training_trajectories-500k.sh
   ```

## Training NNPs with GOLF

### Distributed Gradient Calculation with Psi4
To speed up the training, we parallelize DFT computations using several CPU-rich machines. The training of the NNP takes place on the parent machine with a GPU.
1. Set up environment on the GPU machine like in [the first section](#training-the-nnp-baseline)
1. Log in to CPU-rich machines. They must be accessible via `ssh`.
2. Set up environments on CPU-rich machines.
   ```
   # On CPU rich machines
   git clone <repo>
   cd GOLF/scripts
   ./setup_host.sh n_ports ports_range_begin_train ports_range_begin_eval
   ```
   Here, `n_ports` denotes number of workers on a CPU-rich machine, and `ports_range_begin_train` and `ports_range_begin_eval` denote the starting port numbers for train workers and eval workers respectively. Train workers process conformations during the training phase and eval workers procces conformations during the evaluation phase. For example, `./setup_host.sh 24 20000 30000` will launch a total of 48 workers listening to ports `20000, ... , 20023, 30000, ... , 30023`. Note that train workers and eval workers do not operate at the same time.
   
   By default we assume that each worker uses 4 CPU-cores (can be changed in `env/dft_worker.py`, line 22) which means that `n_ports` must be less or equal to `total_cpu_number / 4`.
4. Add ip addresses of CPU rich machines to a text file. We use `env/host_names.txt`.

### Training with GOLF
Train PaiNN with GOLF.
```
cd scripts/train
./run_training_GOLF.sh
```

## Evaluating NNPs
1. Download evaluation dataset $\mathcal{D}_{\text{test}}$.
   ```
   wget https://n-usr-31b1j.s3pd12.sbercloud.ru/b-usr-31b1j-qz9/data/energy_dbs/GOLF_test.db
   ```
2. Run evaluation.
      
   To optimize conformations with an NNP run the following command. The NNP is specified by providing the path to the experiment folder created during the training. Hyperparameters for the external optimizer and the NNP will be taken from `*path-to-experiment-folder*/config.json`. Specify the checkpoint path _relative_ to the `*path-to-experiment-folder*`. If you whish to run evaluation for a pre-trained NNP, create a folder with `config.json` and an NNP checkpoint.
   ```
   python evaluate_batch_dft.py --checkpoint_path *path-to-experiment-folder* --agent_path *path-to-checkpoint* --n_parallel 240 --n_threads 24 --conf_number -1 --eval_db_path *path_to_evaluation_database* --timelimit 100 --terminate_on_negative_reward False --reward dft --minimize_on_every_step False
   ```
4. Run evaluation and estimate the percentage of optimized energy.
   
   To estimate the $\overline{pct}$, optimal energies obtained with the genuine oracle $\mathcal{O}$ (we used `psi4.optimize`) must be available. We provide them in the evaluation dataset $\mathcal{D}_{\text{test}}$. If only $\overline{pct}_T$ needs to be estimated, run:
   ```
   python evaluate_batch_dft.py --checkpoint_path *path-to-experiment-folder* --agent_path *path-to-checkpoint* --n_parallel 240 --n_threads 24 --conf_number -1 --eval_db_path *path_to_evaluation_database* --timelimit 100 --terminate_on_negative_reward False --reward dft --minimize_on_every_step False --eval_early_stop_steps 100
   ```
   If you wish to adjust the timelimit, adjust `eval_early_stop_steps` accordingly. If you wish to reproduce plots from 1, run:
   ```
   python evaluate_batch_dft.py --checkpoint_path *path-to-experiment-folder* --agent_path *path-to-checkpoint* --n_parallel 240 --n_threads 24 --conf_number -1 --eval_db_path *path_to_evaluation_database* --timelimit 100 --terminate_on_negative_reward False --reward dft --minimize_on_every_step False --eval_early_stop_steps 1 2 3 5 8 13 21 30 50 75 100
   ```
   After the evaluation is finished, an `evaluation_metrics.json` file with per-step metrics is created. Each record in `evaluation_metrics.json` describes optimization statistics for a single conformation and contains such metrics as: forces/energies MSE, percentage of optimized energy, predicted and ground-truth energies, etc. The final NNP-optimized conformations are stored in `results.db` database.

