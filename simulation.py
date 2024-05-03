from ai import *

import time

# File for running the simulation commands

# data = json.load(open("data/1.txt", 'r'))

Manager = DataManager()
# create_network(Manager)

NN = load_best_network()
pygame.init()
self_play_loop(NN)

# play_game()
# training_loop(Manager, NN)