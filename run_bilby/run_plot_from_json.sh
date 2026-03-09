#!/bin/bash 

source /home/hayato.imafuku/research/env/bin/activate

# Run the Python script 
python3 plot_from_json.py -1 "${1}"

# Deactivate environment 
deactivate
