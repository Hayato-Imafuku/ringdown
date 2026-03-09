#!/bin/bash 

# To enable using canda
source /home/hayato.imafuku/miniconda3/etc/profile.d/conda.sh 

# source /home/hayato.imafuku/research/env/bin/activate
conda activate env

# export PYTHONPATH="/home/hayato.imafuku/research/env/lib/python3.9/site-packages:$PYTHONPATH"
# export PYTHONPATH="/home/hayato.imafuku/research/env/lib/python3.9/site-packages/bilby:$PYTHONPATH"

# Run pyring 
pyRing --config-file "${1}"

# Run plot corner
python plot_corner_pyring.py -1 "${1}"

# Deactivate environment 
conda deactivate
