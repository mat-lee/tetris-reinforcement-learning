from ai import *

import numpy as np
import time

### File for running the simulation commands




# Create a network if none exist already

# DefaultConfig = Config(0.5, 16, 1, 16, 16, 10, 0.001)

# if highest_model_ver() == -1:
#     create_network(DefaultConfig, save_network=True, plot_model=False)

# self_play_loop(show_games=True)


# load_best_network().summary()




### Debugging





#data = load_data(0, 10)
# print(len(data))

## Grid search battling different parameters
## 0.5 dropout was best
## 16 neurons was best

'''
learning_rates = [0.001, 0.01, 0.1]
# configs = [Config(0.5, 16, 4, 16, 16, 10, lr) for lr in learning_rates]
configs = [Config(0.5, 16, 4, 16, 16, 10, lr) for lr in learning_rates]
networks = [create_network(config, show_summary=False, save_network=False, plot_model=False) for config in configs]
for network in networks:
    train_network(network, data)

scores={title: {} for title in learning_rates}

pygame.init()

for i in range(len(learning_rates)):
    first_network = networks[i]
    for j in range(i):
        if i != j:
            second_network = networks[j]
            score_1, score_2 = battle_networks_win_loss(first_network, second_network, 
                            network_1_title=learning_rates[i], 
                            network_2_title=learning_rates[j], 
                            show_game=True)
            
            scores[learning_rates[i]][learning_rates[j]] = f"{score_1}-{score_2}"
            scores[learning_rates[j]][learning_rates[i]] = f"{score_2}-{score_1}"


print(scores)'''


'''
neurons = [1, 4, 16, 64, 256, 1024]
neuron_networks = []
neuron_scores={neuron: {} for neuron in neurons}

for neuron in neurons:
    network = create_network(save_network=False, plot_model=False, dropout=0.5, neurons=neuron, layers=10)
    train_network(network, data)
    neuron_networks.append(network)

pygame.init()

for i in range(len(neurons)):
    first_network = neuron_networks[i]
    for j in range(i):
        if i != j:
            second_network = neuron_networks[j]
            score_1, score_2 = battle_networks_win_loss(first_network, second_network, 0.5, 
                            network_1_title=neurons[i], 
                            network_2_title=neurons[j], 
                            show_game=True)
            
            neuron_scores[neurons[i]][neurons[j]] = f"{score_1}-{score_2}"
            neuron_scores[neurons[j]][neurons[i]] = f"{score_2}-{score_1}"



layers = [1, 2, 4, 8, 16, 32]
layer_networks = []
layer_scores={layer: {} for layer in layers}

for layer in layers:
    network = create_network(save_network=False, plot_model=False, dropout=0.5, neurons=1, layers=layer)
    train_network(network, data)
    layer_networks.append(network)

pygame.init()

for i in range(len(layers)):
    first_network = layer_networks[i]
    for j in range(i):
        if i != j:
            second_network = layer_networks[j]
            score_1, score_2 = battle_networks_win_loss(first_network, second_network, 0.5, 
                            network_1_title=layers[i], 
                            network_2_title=layers[j], 
                            show_game=True)
            
            layer_scores[layers[i]][layers[j]] = f"{score_1}-{score_2}"
            layer_scores[layers[j]][layers[i]] = f"{score_2}-{score_1}"'''




# print(dropout_scores)
# print(neuron_scores)
# print(layer_scores)






### The time a model takes to evaluate barely changes with number of parameters
### Good sign: Either I can use a very large network, or I can optimzie tf (i.e. eagerly)

# print(tf.executing_eagerly())

'''data = load_data(20)

X = []

for feature in data[0][:-2]:
    X.append(np.expand_dims(np.array(feature), axis=0))

layers = [2**x for x in range(14)]
times = []

for layer_size in layers:
    Manager = DataManager(residual_layer_size=layer_size)
    NN = create_network(Manager)
    train_network(NN, data)

    START = time.time()
    values, policies = NN(X)
    END = time.time()
    times.append([layer_size, END-START, NN.count_params()])

print(times)'''





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