# Active Learning for Molecular Conformation Optimization
<p align="left">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## Distributed Gradiant Calculation using DFT

1. Prepare an environment and copy to CPU-rich machines. Use the following command
to copy symlinks as symlinks:

   rsync -rlkP LOCAL_DIR REMOTE_HOST:REMOTE_HOST


2. Log in to CPU-rich machines, activate your environment and run babysit_dft.sh. It
launches 16 processes listening to different ports.


3. Modify calculate_dft_energy_queue() in dft.py:161 to connect to those machines:

    for host in ["cpu_machine_1_ip", "cpu_machine_2_ip" ...]:
        for port in range(20000, 20016):


4. Run training on GPU machine as usual.
