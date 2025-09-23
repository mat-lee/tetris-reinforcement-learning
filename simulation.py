import logging
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Suppress Python warnings
warnings.filterwarnings('ignore')

from ai import *

### File for running the simulation commands
show_training = True
skip_first_set = False # If you want to jump straight to model training/evaluation

DefaultConfig = Config(visual=show_training)

# Create a network if none exist already
if highest_model_number(DefaultConfig) == -1:
    instantiate_network(DefaultConfig, show_summary=True, save_network=True, plot_model=False)

# Initiate selfplay
self_play_loop(DefaultConfig, skip_first_set=skip_first_set)

# Command
"/Users/matthewlee/Documents/Code/Tetris Game/SRC/.venv/bin/python" "/Users/matthewlee/Documents/Code/Tetris Game/src/simulation.py"

# Removing hidden ds_store files:
# find . -name ".DS_Store" -delete 