from ai import *

import numpy as np
import time

# File for running the simulation commands

# data = json.load(open("data/1.txt", 'r'))

Manager = DataManager()

create_network(Manager)
NN = load_best_network()

self_play_loop(NN, show_games=True)
# pygame.init()
# battle_networks(NN, NN, show_game=True)

# play_game(NN, 0, show_game=True)
# training_loop(Manager, NN)



### Debugging

'''pygame.init()
screen = pygame.display.set_mode( (WIDTH, HEIGHT))
pygame.display.set_caption(f'Debugging')

NN = load_best_network()

game = Game()
game.setup()

move, tree = MCTS(game, NN)
search_matrix = search_statistics(tree)

for policy, move in get_move_list(search_matrix, np.ones((2,25,11,4))):
    game.make_move(move)
    game.show(screen)
    pygame.display.update()
    #time.sleep(1)
    game.undo()'''

# network = keras.models.clone_model(NN)
# network.summary()

'''data = json.load(open(f"{directory_path}/1.4.1.txt", 'r'))[30:31]
X = []
for feature in data[0][:-2]:
    X.append(np.expand_dims(np.array(feature), axis=0))

train_network(network, data)

values, policies = network.predict(X, verbose=0)
policies = np.array(policies)
policies = policies.reshape((2, ROWS, COLS+1, 4))

diff = policies - np.array(data[0][-1])

print("a")'''