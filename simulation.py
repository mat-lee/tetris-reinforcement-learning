from ai import *

import time

#make_traning_set(load_best_network(), 1)

'''    
    data = json.load(open("data/nohold/1-1game-10depth.txt", 'r'))
    create_network(data)'''

# create_network()
NN = load_best_network()
# NN.summary()
self_play_loop(NN)