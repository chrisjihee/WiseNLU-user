#!/bin/bash
# conda
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# basic
mamba create -n WiseNLU-user python=3.11 -y
mamba activate WiseNLU-user
pip install -r requirements.txt

# chrisbase
rm -rf chrisbase*
git clone https://github.com/chrisjihee/chrisbase.git
pip install --editable chrisbase*

# list
pip list | grep -E "chris"
