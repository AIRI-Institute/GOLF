# Active Learning for Molecular Conformation Optimization
<p align="left">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## Distributed Gradiant Calculation using DFT
1. Set up environments on CPU-rich machines.
```
# On CPU rich machines
git clone <repo>
cd GOLF/scripts
./setup_host.sh n_ports ports_range_begin_train ports_range_begin_eval
```
Here, `n_ports` denotes number of workers on a CPU-rich machine, and `ports_range_begin_train` and `ports_range_begin_eval` denote the starting port numbers for train workers and eval workers respectively.Train workers process conformations during the training phase and eval workers procces conformations during the evaluation phase. For example, `./setup_host.sh 24 20000 30000` will launch a total of 48 workers on ports $20000, \dots, 20023, 30000, \dots, 30023$.

By default we assume that each worker uses 4 CPU-cores (can be changed in `env/dft_worker.py`, line 22) which means that `n_ports` must be less or equal `total_cpu_number / 4`.

1. Prepare an environment and copy to CPU-rich machines. Use the following command
to copy symlinks as symlinks:

   rsync -rlkP LOCAL_DIR REMOTE_HOST:REMOTE_HOST


2. Log in to CPU-rich machines, activate your environment and run babysit_dft.sh. It
launches 16 processes listening to different ports.


3. Modify calculate_dft_energy_queue() in dft.py:161 to connect to those machines:

    for host in ["cpu_machine_1_ip", "cpu_machine_2_ip" ...]:
        for port in range(20000, 20016):


4. Run training on GPU machine as usual.
