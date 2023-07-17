#! /bin/bash -ex

# kill `ps uaxww | grep dft.py | awk '{ print $2 }'`

function cleanup {
    echo "stop workers"
    pkill -P $$

    echo "clean /dev/shm"
    rm -f "/dev/shm/psi.*"
}

trap cleanup EXIT

range=$1
begin=$2
end=`expr $begin + $range`

rm -f /dev/shm/psi.*

for port in `seq $begin $end`;
do
    python3.9 ../env/dft.py $port &
done

wait