#! /bin/bash -ex

num_threads=$1
n_ports=$2
begin_range_train=$3
./install_miniconda.sh
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
./install_env.sh
conda activate golf_dft_env
./babysit_dft.sh $num_threads $n_ports $begin_range_train

