#! /bin/sh -ex

# kill `ps uaxww | grep dft.py | awk '{ print $2 }'`

trap 'kill -9 $(jobs -p)' EXIT

rm -f /dev/shm/psi.*

for port in `seq 20000 20015`;
do
    python3.9 ./env/dft.py $port &
done

wait
