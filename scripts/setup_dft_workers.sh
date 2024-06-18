#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to print usage
print_usage() {
    echo "Usage: $0 [-r|--relaunch] <num_threads> <num_workers> <start_port>"
}

# Parse arguments
RELAUNCH_ONLY=0
if [[ "$1" == "-r" || "$1" == "--relaunch" ]]; then
    RELAUNCH_ONLY=1
    shift
fi

# Ensure the correct number of arguments are provided
if [[ $# -ne 3 ]]; then
    print_usage
    exit 1
fi

# Parameters
NUM_THREADS=$1
NUM_WORKERS=$2
START_PORT=$3

# Function to install Mamba
install_mamba() {
    echo "Installing Mamba..."
    mkdir -p ~/mamba3
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O ~/mamba3/mamba.sh
    bash ~/mamba3/mamba.sh -b -u -p ~/mamba3
    rm -rf ~/mamba3/mamba.sh
    ~/mamba3/bin/conda init bash
}

# Function to setup the environment
setup_environment() {
    echo "Setting up the environment..."
    source ~/mamba3/etc/profile.d/conda.sh # Initialize Conda/Mamba environment
    conda create -y -n golf_dft_env python=3.10 # Create the environment with Python 3.10
    conda activate golf_dft_env # Activate the newly created environment
    conda install -y psi4 -c conda-forge # Install psi4 using Mamba
    conda install -y ase -c conda-forge # Install ase using Mamba
}

# Function to activate the environment
activate_environment() {
    echo "Activating the environment..."
    source ~/mamba3/etc/profile.d/conda.sh # Initialize Conda/Mamba environment
    conda activate golf_dft_env # Activate the environment
}

# Function to launch workers
launch_workers() {
    echo "Launching workers..."
    ./babysit_dft.sh $NUM_THREADS $NUM_WORKERS $START_PORT
}

# Check if Mamba is already installed
if ! command -v mamba &> /dev/null; then
    install_mamba
else
    echo "Mamba is already installed."
fi

# Main script execution
if [[ $RELAUNCH_ONLY -eq 0 ]]; then
    setup_environment
else
    activate_environment
fi
launch_workers