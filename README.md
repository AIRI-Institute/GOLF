# Active Learning for Molecular Conformation Optimization
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
   ./run_training_baseline
   ```

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
   Running these scripts will create a folder in the specified `log_dir` directory. The name of the folder is specified by the `exp_name` hyperparameter. The folder will contain checkpoints, a metrics file and a config file with hyperparameters.

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
4. Write ip addresses of CPU rich machines to a text file. We use `env/host_names.txt`.

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
   ```
   python evaluate_batch_dft.py --checkpoint_path *path-to-experiment-folder* --agent_path *path-to-checkpoint* --n_parallel 120 --n_threads 24 --conf_number -1 --eval_db_path *path_to_evaluation_database*
   ```


