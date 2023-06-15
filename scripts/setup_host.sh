#! /bin/bash -ex

n_ports=$1
./install_miniconda.sh
source ~/miniconda3/etc/profile.d/conda.sh
./install_env.sh
conda activate golf_dft_env
./babysit_dft.sh $n_ports
