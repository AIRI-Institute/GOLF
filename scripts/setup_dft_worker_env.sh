#!/bin/bash -ex

conda create -y -n golf_dft_env python=3.12
conda install -y -n golf_dft_env psi4 -c conda-forge 
conda install -y -n golf_dft_env ase -c conda-forge

