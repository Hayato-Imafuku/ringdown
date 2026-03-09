#!/bin/bash

# To enable using conda
source /home/hayato.imafuku/miniconda3/etc/profile.d/conda.sh 

conda activate env

pip install -e /home/hayato.imafuku/Development/pyring/

# conda deactivate