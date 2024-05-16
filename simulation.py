from ai import *

import numpy as np
import time

# File for running the simulation commands
'''Manager = DataManager()

create_network(Manager)
NN = load_best_network()

self_play_loop(NN, show_games=True)'''



### Debugging
data = load_data(20)

X = []

for feature in data[0][:-2]:
    X.append(np.expand_dims(np.array(feature), axis=0))

layers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
times = []

for layer_size in layers:
    Manager = DataManager(residual_layer_size=layer_size)
    create_network(Manager)
    NN = load_best_network()
    train_network(NN, data)

    START = time.time()
    values, policies = NN(X)
    END = time.time()
    times.append([layer_size, END-START])

fig, ax = plt.subplots()
ax.plot(times)

print(times)




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