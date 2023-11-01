#!/bin/bash -ex

conda create -y -n golf_dft_env python=3.9
conda install -y -n golf_dft_env psi4 -c psi4
conda install -y -n golf_dft_env ase -c conda-forge

