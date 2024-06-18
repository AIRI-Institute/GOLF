#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to clean up resources
cleanup() {
    echo "Stopping workers..."
    pkill -P $$

    echo "Cleaning shared memory..."
    rm -f /dev/shm/psi* /dev/shm/null* /dev/shm/dfh*
}

# Trap EXIT signal to clean up resources
trap cleanup EXIT

# Parameters
NUM_THREADS=$1
PORT_RANGE=$2
START_PORT=$3
END_PORT=$(($START_PORT + $PORT_RANGE - 1))

# Clean up any leftover shared memory files
rm -f /dev/shm/psi* /dev/shm/null* /dev/shm/dfh*

# Launch workers
for PORT in $(seq $START_PORT $END_PORT); do
    python ../env/dft.py $NUM_THREADS $PORT &> worker_$PORT.out &
done

# Wait for all background jobs to finish
wait