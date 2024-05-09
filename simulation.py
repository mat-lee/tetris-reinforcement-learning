from ai import *

import time

# File for running the simulation commands

# data = json.load(open("data/1.txt", 'r'))

Manager = DataManager()

create_network(Manager)
NN = load_best_network()

self_play_loop(NN, manager=Manager, show_games=True)

# play_game()
# training_loop(Manager, NN)



### Debugging

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