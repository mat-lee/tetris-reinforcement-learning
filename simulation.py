from ai import *

### File for running the simulation commands

DefaultConfig = Config()
# Create a network if none exist already
if highest_model_ver(CURRENT_VERSION) == -1:
    instantiate_network(DefaultConfig, generate_fishlike_network, save_network=True, plot_model=False)

# Initiate selfplay
self_play_loop(DefaultConfig, skip_first_set=False, show_games=True)