from ai import *

### File for running the simulation commands

DefaultConfig = Config()

# Create a network if none exist already
if highest_model_number(MODEL_VERSION) == -1:
    instantiate_network(DefaultConfig, nn_generator=gen_alphasame_nn, show_summary=True, save_network=True, plot_model=False)

# Initiate selfplay
self_play_loop(DefaultConfig, skip_first_set=False, show_games=True)