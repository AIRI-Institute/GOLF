# Active Learning for Molecular Conformation Optimization
<p align="left">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## Distributed Gradient Calculation with Psi4
To speed up the training, we parallelize DFT computations using several CPU-rich machines. The training of the NNP takes place on the parent machine with a GPU.
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
3. Set up environment on the GPU machine.
   ```
   # On the GPU machine
   conda create -y -n md_env python=3.9
   conda activate md_env
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install  psi4 -c psi4
   python -m pip install -r requirements.txt
   ```
4. Write ip addresses of CPU rich machines to a text file. We use `env/host_names.txt`.
5. 

## Training NNPs on optimization trajectories
1. Set up environment on the GPU machine.
   ```
   # On the GPU machine
   conda create -y -n md_env python=3.9
   conda activate md_env
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install  psi4 -c psi4
   python -m pip install -r requirements.txt
   ```
2. Download trajectories.
   ```
   wget https://n-usr-31b1j.s3pd12.sbercloud.ru/b-usr-31b1j-qz9/data/energy_dbs/traj-10k.db
   wget https://n-usr-31b1j.s3pd12.sbercloud.ru/b-usr-31b1j-qz9/data/energy_dbs/traj-100k.db
   wget https://n-usr-31b1j.s3pd12.sbercloud.ru/b-usr-31b1j-qz9/data/energy_dbs/traj-500k.db
   ```
3. Train PaiNN.
   ```
   ./run_training_trajectories.sh
   ```
