#! /bin/bash -ex

# kill `ps uaxww | grep dft.py | awk '{ print $2 }'`

function cleanup {
    echo "stop workers"
    pkill -P $$

    echo "clean /dev/shm"
    rm -f /dev/shm/psi* /dev/shm/null* /dev/shm/dfh*
}

trap cleanup EXIT

num_threads=$1
range=$2
begin=$3
end=`expr $begin + $range - 1`

rm -f /dev/shm/psi* /dev/shm/null* /dev/shm/dfh*

for port in `seq $begin $end`;
do
    python ../env/dft.py $num_threads $port &>worker_$port.out &
done

wait
