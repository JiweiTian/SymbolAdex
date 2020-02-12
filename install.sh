#!/bin/bash
git clone https://github.com/eth-sri/ERAN.git
cd ERAN
git checkout 73f48315ae28da1a21f90ac9a6ef43338dd3190b
sudo ./install.sh
source gurobi_setup_path.sh
cd ..
cp ./patches/clever_wolf_main.py ./ERAN/tf_verify/
cp ./patches/clever_wolf.py ./ERAN/tf_verify/
cp ./patches/clever_pgd_generator.py ./ERAN/tf_verify/
cp -r ./patches/nets/ ./ERAN/
cp ./patches/scripts/* ./ERAN/
patch ./ERAN/tf_verify/deepzono_milp.py ./patches/patch_deepzono_milp.patch
patch ./ERAN/tf_verify/read_net_file.py ./patches/patch_readnet_file.patch
pip3 install -r requirements.txt
