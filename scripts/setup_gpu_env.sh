conda create -y -n GOLF_pyg python=3.10
conda activate GOLF_pyg
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg -c pyg
conda install  psi4 -c psi4
python -m pip install -r requirements.txt