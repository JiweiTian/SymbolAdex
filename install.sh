#!/bin/bash
git clone https://github.com/eth-sri/ERAN.git
cd ERAN
git checkout 73f48315ae28da1a21f90ac9a6ef43338dd3190b
sudo ./install.sh
source gurobi_setup_path.sh
pip3 install -r requirements.txt
cd ..
cp ./patches/clever_wolf_main.py ./ERAN/tf_verify/
cp ./patches/clever_wolf.py ./ERAN/tf_verify/
cp ./patches/clever_pgd_generator.py ./ERAN/tf_verify/
pip3 install -r requirements.txt
