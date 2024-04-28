from ai import *

import time

#make_traning_set(load_best_network(), 1)

'''    
    data = json.load(open("data/nohold/1-1game-10depth.txt", 'r'))
    create_network(data)'''

START = time.time()
make_traning_set(load_best_network(), 1)
END = time.time()
print(END - START)