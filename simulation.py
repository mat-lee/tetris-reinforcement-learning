from ai import *

### File for running the simulation commands
show_training = True

DefaultConfig = Config()

# Create a network if none exist already
if highest_model_number(DefaultConfig) == -1:
    instantiate_network(DefaultConfig, show_summary=True, save_network=True, plot_model=False)

# Initiate selfplay
self_play_loop(DefaultConfig, skip_first_set=False)

# Command
"/Users/matthewlee/Documents/Code/Tetris Game/SRC/.venv/bin/python" "/Users/matthewlee/Documents/Code/Tetris Game/src/simulation.py"

# Removing hidden ds_store files:
# find . -name ".DS_Store" -delete 