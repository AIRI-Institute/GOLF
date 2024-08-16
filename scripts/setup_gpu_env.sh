conda create -y -n GOLF_schnetpack python=3.12
conda install -y -n GOLF_schnetpack pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y -n GOLF_schnetpack lightning -c conda-forge
conda install -y -n GOLF_schnetpack psi4 -c conda-forge
conda install -y -n GOLF_schnetpack rdkit -c conda-forge