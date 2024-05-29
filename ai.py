from const import *
from game import Game

import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import treelib

import keras
import tensorflow as tf
from tensorflow.python.ops import math_ops
from sklearn.model_selection import train_test_split

import cProfile
import pstats

# For naming data and models
CURRENT_VERSION = 3.7

# Tensorflow settings to use eager execution
# Had the same performance
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_v2_behavior()
#tf.config.run_functions_eagerly(False)

# Where data and models are saved
directory_path = '/Users/matthewlee/Documents/Code/Tetris Game/Storage'

# Areas of optimization:
# - Finding piece locations (piece locations held and not held are probably related)
# - Optimizing piece coordinates
# - Piece rotations that result in the same board (Pick random one)

# AI todo:
# - Improve network (e.g. L2, CNNs, adjust number of weights/layers, Dropout)
# - Adjust parameters (CPuct, Dirichlet noise, Temperature)
# - Encoding garbage into the neural network
# - Data augmentation

total_branch = 0
number_branch = 0

class NodeState():
    """Node class for storing the game in the tree.
    
    Similar to the game class, but is more specialized for AI functions and has different variables"""

    def __init__(self, game=None, move=None) -> None:
        self.game = game
        self.move = move

        # Save garbage statistics instead of spawning garbage
        # Each list index corressponds to some amount of garbage
        # spawning when placing that piece
        self.garbage = [[0] * 7, [0] * 7]

        # Search tree variables
        # Would be stored in each edge, but this is essentially the same thing
        self.visit_count = 1
        self.value_sum = 0
        self.value_avg = 0
        self.policy = 0

def MCTS(game, network, add_noise=False):
    global total_branch, number_branch
    # Picks a move for the AI to make 
    # Initialize the search tree
    tree = treelib.Tree()
    game_copy = game.copy()

    # Orient the game so that the AI is active
    if game_copy.turn == 1:
        game_copy.players = game_copy.players[::-1]
        game_copy.turn = 0

    # Restrict previews
    for player in game_copy.players:
        while len(player.queue.pieces) > PREVIEWS:
            player.queue.pieces.pop(-1)

    # Create the initial node
    initial_state = NodeState(game=game_copy, move=None)

    tree.create_node(identifier="root", data=initial_state)

    MAX_DEPTH = 0
    iter = 0

    while iter < MAX_ITER:
        iter += 1

        # Begin at the root node
        node = tree.get_node("root")
        node_state = node.data

        DEPTH = 0

        # Go down the tree using formula Q+U until you get to a leaf node
        # Looking at promising moves for both players
        while not node.is_leaf():
            child_ids = node.successors(tree.identifier)
            max_child_score = -1
            max_child_id = None
            parent_visits = node.data.visit_count

            number_branch += 1
            
            Qs = []
            Us = []

            for child_id in child_ids:
                
                total_branch += 1

                # For each child calculate a score
                # Polynomial upper confidence trees (PUCT)
                child_data = tree.get_node(child_id).data
                child_score = child_data.value_avg + CPUCT * child_data.policy*math.sqrt(parent_visits)/(1+child_data.visit_count)

                Qs.append(child_data.value_avg)
                Us.append(child_data.policy*math.sqrt(parent_visits)/(1+child_data.visit_count))

                if child_score >= max_child_score:
                    max_child_score = child_score
                    max_child_id = child_id
            
            # if iter == 10:
            #     fig, axs = plt.subplots(3)
            #     fig.suptitle('Q (value) vs U (policy) vs Q+U')
            #     axs[0].plot(Qs)
            #     axs[1].plot(Us)
            #     axs[2].plot(np.array(Qs)+np.array(Us))
            #     plt.savefig(f"{directory_path}/UCB_model_3_6_4_depth_{iter}")
            #     print("saved")
            
            # Pick the node with the highest score
            node = tree.get_node(max_child_id)
            node_state = node.data

            DEPTH += 1
            if DEPTH > MAX_DEPTH:
                MAX_DEPTH = DEPTH

        # If not the root node, place piece in node
        if not node.is_root():
            prior_node = tree.get_node(node.predecessor(tree.identifier))
            
            game_copy = prior_node.data.game.copy()
            node_state.game = game_copy
            node_state.game.make_move(node_state.move, add_bag=False, add_history=False)

        # Don't update policy, move_list, or generate new nodes if the game is over       
        if node_state.game.is_terminal == False:
            value, policy = evaluate(node_state.game, network)
            # value, policy = random_evaluate()

            if node_state.game.no_move == False:
                move_matrix = get_move_matrix(node_state.game.players[node_state.game.turn])
                move_list = get_move_list(move_matrix, policy)

                # Generate new leaves
                # Normalize policy values
                policy_sum = sum(policy for policy, move in move_list)

                for policy, move in move_list:
                    new_state = NodeState(game=None, move=move)
                    new_state.policy = policy / policy_sum

                    tree.create_node(data=new_state, parent=node.identifier)
        
        # Node is terminal
        # Update weights based on winner
        # Range values from 0 to 1 so that value avg will be 0.5
        else: 
            winner = node_state.game.winner
            if winner == 0: # AI Starts
                value = 1
            elif winner == 1: # Opponent
                value = 0
            else: # Draw
                value = 0.5

        # If root node, add exploration noise to children
        if (add_noise == True and node.is_root()):
            child_ids = node.successors(tree.identifier)
            number_of_children = len(child_ids)
            noise_distribution = np.random.gamma(DIRICHLET_ALPHA, 1, number_of_children)

            pre_noise_policy = []
            post_noise_policy = []

            for child_id, noise in zip(child_ids, noise_distribution):
                child_data = tree.get_node(child_id).data
                pre_noise_policy.append(child_data.policy)

                child_data.policy = child_data.policy * (1 - DIRICHLET_EXPLORATION) + noise * DIRICHLET_EXPLORATION
                post_noise_policy.append(child_data.policy)
            
            # fig, axs = plt.subplots(2)
            # fig.suptitle('Policy before and after dirichlet noise')
            # axs[0].plot(pre_noise_policy)
            # axs[1].plot(post_noise_policy)
            # plt.savefig(f"{directory_path}/policy_4_2_1")
            # print("saved")

        # Go back up the tree and updates nodes
        # Propogate positive values for the player made the move, and negative for the other player
        final_node_turn = node_state.game.turn

        while not node.is_root():
            node_state = node.data
            node_state.visit_count += 1
            # Revert value if the other player just went
            node_state.value_sum += (value if node_state.game.turn == final_node_turn else 1-value)
            node_state.value_avg = node_state.value_sum / node_state.visit_count

            upwards_id = node.predecessor(tree.identifier)
            node = tree.get_node(upwards_id)

        # Repeat for root node
        node_state = node.data
        node_state.visit_count += 1
        node_state.value_sum += (value if node_state.game.turn == final_node_turn else 1-value)
        node_state.value_avg = node_state.value_sum / node_state.visit_count

    #print(MAX_DEPTH, total_branch//number_branch)

    # Choose a move based on the number of visits
    root = tree.get_node("root")
    root_children_id = root.successors(tree.identifier)
    max_n = 0
    max_id = None

    root_child_n_list = []

    for root_child_id in root_children_id:
        root_child = tree.get_node(root_child_id)
        root_child_n = root_child.data.visit_count
        root_child_n_list.append(root_child_n)
        if root_child_n > max_n:
            max_n = root_child_n
            max_id = root_child.identifier

    # fig, axs = plt.subplots(2)
    # fig.suptitle('N Compared to policy')
    # axs[0].plot(post_noise_policy)
    # axs[1].plot(root_child_n_list)
    # plt.savefig(f"{directory_path}/root_n_{MAX_ITER}_depth_3_4_2")
    # print("saved")

    move = tree.get_node(max_id).data.move

    return move, tree

def get_move_matrix(player):
    # Returns a list of all possible moves that a player can make
    # Very Important: With my coordinate system, pieces can be placed with negative x values
    # To avoid negative indexing, the list of moves is shifted by 2
    # It encodes moves from -2 to 8 as indices 0 to 10

    # Possible locations needs to be larger to acount for blocks with negative x and y
    # O piece        0 to 8
    # Any 3x3 piece -1 to 8
    # I piece       -2 to 8

    def get_highest_row(grid):
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] != 0:
                    highest_row = row
                    return highest_row
        
        # Else return the floor
        return len(grid)

    # On the left hand side, blocks can have negative x
    # buffer = (2 if piece.type == "I" else (0 if piece.type == "O" else 1))
    buffer = 2
    width = COLS + buffer - 1

    # Convert to new policy format (19, 25 x 11)
    new_policy = np.zeros(POLICY_SHAPE)

    # Load the state
    sim_player = player.copy()
    piece_1 = None
    piece_2 = None

    for h in range(2): # Look through current piece and held piece
        if h == 0:
            if sim_player.piece != None:
                piece_1 = sim_player.piece.type
        if h == 1:
            sim_player.hold_piece()
            if sim_player.piece != None:
                piece_2 = sim_player.piece.type

        piece = sim_player.piece

        if piece != None and piece_1 != piece_2: # Skip if there's no piece or the pieces are the same
            possible_piece_locations = np.zeros((ROWS, width, 4))

            # Queue for looking through piece placements
            next_location_queue = []

            # Start the piece at the highest point it can be placed
            highest_row = get_highest_row(sim_player.board.grid)
            starting_row = max(highest_row - len(piece_dict[piece.type]), ROWS - SPAWN_ROW)
            piece.y_0 = starting_row

            next_location_queue.append((piece.x_0, piece.y_0, piece.rotation))

            # Search through the queue
            while len(next_location_queue) > 0:
                piece.x_0, piece.y_0, piece.rotation = next_location_queue[0]

                piece.coordinates = piece.get_self_coords

                possible_piece_locations[piece.y_0][piece.x_0 + buffer][piece.rotation] = 1

                # Check left, right, and down moves
                for move in [[1, 0], [-1, 0], [0, 1]]:
                    if sim_player.can_move(piece, x_offset=move[0], y_offset=move[1]): # Check this first to avoid index errors
                        x = piece.x_0 + move[0]
                        y = piece.y_0 + move[1]
                        o = piece.rotation

                        if (possible_piece_locations[y][x + buffer][o] == 0 
                            and (x, y, o) not in next_location_queue
                            and y >= 0): # Avoid negative indexing

                            next_location_queue.append((x, y, o))

                # Check rotations 1, 2, and 3
                for i in range(1, 4):
                    sim_player.try_wallkick(i)

                    x = piece.x_0
                    y = piece.y_0
                    o = piece.rotation

                    if (possible_piece_locations[y][x + buffer][o] == 0
                        and (x, y, o) not in next_location_queue
                        and y >= 0): # Avoid negative indexing
                        next_location_queue.append((x, y, o))

                    # Reset piece locations
                    piece.x_0, piece.y_0, piece.rotation = next_location_queue[0]

                next_location_queue.pop(0)

            # Remove entries that can move downwards
            for o in range(4): # Smallest number of operations
                sim_player.piece.rotation = o
                rotation_index = o % len(policy_pieces[piece.type])
                policy_index = policy_piece_to_index[piece.type][rotation_index]

                for x in range(COLS + buffer - 1):
                    sim_player.piece.x_0 = x - buffer

                    for y in reversed(range(ROWS - 1)):
                        if possible_piece_locations[y][x][o] == 1:
                            sim_player.piece.y_0 = y
                            piece.coordinates = piece.get_self_coords
                            if not sim_player.can_move(piece, y_offset=1): # If can't move down
                                new_col = x
                                new_row = y
                                if piece.type in ["Z", "S", "I"]:
                                    # For those pieces, rotation 2 is the same as rotation 0
                                    # but moved one down
                                    if o == 2:
                                        new_row += 1
                                    # For those pieces, rotation 3 is the same as rotation 1
                                    # but moved one to the left
                                    if o == 3:
                                        new_col -= 1
                                new_policy[policy_index][new_row][new_col] = 1
        
    return new_policy

def get_move_list(move_matrix, policy_matrix):
    # Returns list of possible moves with their policy
    # Removes buffer
    move_list = []

    mask = np.multiply(move_matrix, policy_matrix)
    # Ordered from smallest to largest
    for layer in range(len(mask)):
        piece, rotation = policy_index_to_piece[layer]
        for col in range(len(mask[0][0])):
            for row in range(len(mask[0])):
                if mask[layer][row][col] > 0:
                    move_list.append((mask[layer][row][col],
                                        (piece, col - 2, row, rotation) # Switch to x y z and remove buffer
                                        ))
    # Sort by policy
    # move_list = sorted(move_list, key=lambda tup: tup[0], reverse=True)
    return move_list

# Using deepcopy:                       100 iter in 36.911 s
# Using copy functions in classes:      100 iter in 1.658 s
# Many small changes:                   100 iter in 1.233 s
# MCTS uses game instead of player:     100 iter in 1.577 s
# Added large NN but optimized MCTS:    100 iter in 7.939 s
#   Without NN:                         100 iter in 0.882 s
#   Changed collision and added coords: 100 iter in 0.713 s
# Use Model(X) instead of .predict:     100 iter in 3.506 s
# Use Model.predict_on_batch(X):        100 iter in 1.788 s

##### Neural Network #####
# 
# Player orientation: Active player, other player
# y:
#   Policy: (2 x 25 x 11 x 4) = 2200 (Hold x Rows x Columns x Rotations)
#   Value: (1)
# 
# Model <1.7: In the trenches
# 
# Model 1.8: Added data augmentation and changed amount of training data and added dirichlet noise: 
# Performs semi rare line clears
# 
# Model 1.9: Fixed data augmentation pieces not being swapped correctly
# Back in the trenches
#
# Model 2.0: Normalized policy values, minor piece optimization improvement, major tf optimizations:
# Increased iter, heavily enlarged network (37 million parameters)
# Out of the trenches: some line clears
#
# Model 2.1: Reduced network, (grids -> cnn) + flattened inputs -> dense -> output, added continuous training
# Back into trenches
#
# Model 2.2: Completely restructured network
# Trenches
#
# Model 2.3: Enlargened/lengthened network + saved image, max iter: 40 -> 80, battle games: 20 -> 40
# Some line clears
# 
# Model 2.4: Model only uses data from the current best network
# Awful
#
# Model 2.5: Adjusted dropout to 0.5
# Awful
#
# Model 2.6: Adjusted neurons to 16 and max_iter to 160
# Awful
#

class Config():
    def __init__(self, 
                 dropout, 
                 filters, 
                 single_filters, 
                 first_neurons, 
                 head_neurons, 
                 layers, 
                 learning_rate):
        self.dropout = dropout
        self.filters = filters
        self.single_filters = single_filters
        self.first_neurons = first_neurons
        self.head_neurons = head_neurons
        self.layers = layers
        self.learning_rate = learning_rate

def create_network(config: Config, show_summary=True, save_network=True, plot_model=False):
    # Creates a network with random weights
    # 1: For each grid, apply the same neural network, and then use 1x1 kernel and concatenate
    # 1 -> 2: For opponent grid, apply fully connected layer
    # Concatenate active player's kernels/features with opponent's dense layer and non-player specific features
    # Apply value head and policy head 

    shapes = [(ROWS, COLS, 1), # Grid
                (2 + PREVIEWS, len(MINOS)), # Pieces
                (1,), # B2B
                (1,), # Combo
                (1,), # Lines cleared
                (1,), # Lines sent
                (ROWS, COLS, 1), 
                (2 + PREVIEWS, len(MINOS)), 
                (1,), 
                (1,), 
                (1,),
                (1,),
                (1,), # Color (Whether you had first move or not)
                (1,)] # Total pieces placed

    def create_input_layers(shapes):
        # Use a convolutional layer
        inputs = []
        active_grid = None
        active_features = []
        other_grid = None
        other_features = []
        
        non_player_features = []

        for i, shape in enumerate(shapes):
            # Add input
            input = keras.Input(shape=shape)
            inputs.append(input)

            num_inputs = len(shapes)
            # Active player's features
            if i < (num_inputs - 2) / 2: # Ignore last two inputs, and take the first half
                if shape == shapes[0]:
                    active_grid = input
                else:
                    active_features.append(keras.layers.Flatten()(input))
            # Other player's features
            elif i < (num_inputs - 2): # Ignore last two inputs, take remaining half
                if shape == shapes[0]:
                    other_grid = input
                else:
                    other_features.append(keras.layers.Flatten()(input))
            # Other features
            else:
                non_player_features.append(keras.layers.Flatten()(input))
        
        return inputs, active_grid, active_features, other_grid, other_features, non_player_features
    
    def ResidualConvLayer():
        # Uses skip conections
        def inside(in_1, in_2):
            conv_1 = keras.layers.Conv2D(config.filters, (3, 3), padding="same")
            batch_1 = keras.layers.BatchNormalization()
            relu_1 = keras.layers.Activation('relu')
            conv_2 = keras.layers.Conv2D(config.filters, (3, 3), padding="same")
            batch_2 = keras.layers.BatchNormalization()
            relu_2 = keras.layers.Activation('relu')

            out_1 = relu_2(batch_2(conv_2(relu_1(batch_1(conv_1(in_1))))))
            out_2 = relu_2(batch_2(conv_2(relu_1(batch_1(conv_1(in_2))))))

            out_1 = keras.layers.Add()([in_1, out_1])
            out_2 = keras.layers.Add()([in_2, out_2])

            dropout_1 = keras.layers.Dropout(config.dropout)

            out_1 = dropout_1(out_1)
            out_2 = dropout_1(out_2)

            return out_1, out_2
        return inside

    def ValueHead():
        # Returns value; found at the end of the network
        def inside(x):
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.Dense(config.head_neurons)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.Dropout(config.dropout)(x) # Dropout
            x = keras.layers.Dense(1, activation='sigmoid')(x)

            return x
        return inside

    def PolicyHead():
        # Returns policy list; found at the end of the network
        def inside(x):
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            # Generate probability distribution
            x = keras.layers.Dense(POLICY_SIZE, activation="softmax")(x)

            return x
        return inside

    inputs, active_grid, active_features, other_grid, other_features, non_player_features = create_input_layers(shapes)

    # Start with a convolutional layer
    # Because each grid needs the same network, use the same layers for each side
    conv_1 = keras.layers.Conv2D(config.filters, (3, 3), padding="same")
    batch_1 = keras.layers.BatchNormalization()
    relu_1 = keras.layers.Activation('relu')
    dropout_1 = keras.layers.Dropout(config.dropout)
    
    out_1 = dropout_1(relu_1(batch_1(conv_1(active_grid))))
    out_2 = dropout_1(relu_1(batch_1(conv_1(other_grid))))

    # Use residual blocks
    for _ in range(config.layers):
        residual_layer = ResidualConvLayer()
        out_1, out_2 = residual_layer(out_1, out_2)
    # Use 1x1 kernels
    kernel_1 = keras.layers.Conv2D(config.single_filters, (1, 1))
    out_1 = kernel_1(out_1)
    out_2 = kernel_1(out_2)

    # Flatten layers
    flatten_1 = keras.layers.Flatten()
    out_1 = flatten_1(out_1)
    out_2 = flatten_1(out_2)

    # Concatenate with features
    out_1 = keras.layers.Concatenate()([out_1, *active_features])
    out_2 = keras.layers.Concatenate()([out_2, *other_features])

    # Connect other player with fully connected layer
    out_2 = keras.layers.Dense(config.first_neurons)(out_2)

    # Concatenate with active player's layers and other features
    out = keras.layers.Concatenate()([out_1, out_2, *non_player_features])

    value_output = ValueHead()(out)
    policy_output = PolicyHead()(out)

    model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])

    if plot_model == True:
        keras.utils.plot_model(model, to_file=f"{directory_path}/model_{CURRENT_VERSION}_img.png", show_shapes=True)

    # Loss is the sum of MSE of values and Cross entropy of policies
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate), loss=["mean_squared_error", "categorical_crossentropy"], loss_weights=[1, 1])

    if show_summary: model.summary()

    #model.fit(x=grids, y=[values, policies])

    if save_network: model.save(f"{directory_path}/{CURRENT_VERSION}.0.keras")

    return model

def train_network(model, data):
    # Fit the model
    # Swap rows and columns
    features = list(map(list, zip(*data)))

    # Make features into np arrays
    for i in range(len(features)):
        features[i] = np.array(features[i])

    # Last two columns are value and policy
    policies = features.pop()
    values = features.pop()

    # Reshape policies
    policies = np.array(policies).reshape((-1, POLICY_SIZE))

    # callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience = 20)

    model.fit(x=features, y=[values, policies], batch_size=32, epochs=1, shuffle=True)

def evaluate(game, network):
    # Use a neural network to return value and policy.
    data = game_to_X(game)
    X = []
    for feature in data:
        # expanded_feature = np.expand_dims(np.array(feature), axis=0)
        # X.append(tf.convert_to_tensor(expanded_feature))
        X.append(np.expand_dims(np.array(feature), axis=0))
        
    value, policies = network.predict_on_batch(X)
    # Both value and policies are returned as arrays
    policies = np.array(policies)
    policies = policies.reshape(POLICY_SHAPE)
    return value[0][0], policies.tolist()

    # values, policies = network.predict(X, verbose=0)
    # value, policies = network(X)
    # Convert value from tensor to float
    # value = value.numpy()[0]

def random_evaluate():
    # For testing how fast the MCTS is
    return random.random() * 2 -1, np.random.rand(19, ROWS-1, COLS+1).tolist()

##### Simulation #####

# Having two AI's play against each other to generate a self-play dataset.
# At each move, save X and y for NN

def search_statistics(tree):
    """Return a matrix of proportion of moves looked at.

    Equal to Times Visited / Total Nodes looked at for each move,
    and will become the target for training policy.
    Policy: Rows x Columns x Rotations x Hold
    Policy: 25 x 11 x 4 x 2"""

    # ACCOUNT FOR BUFFER
    probability_matrix = np.zeros(POLICY_SHAPE)

    root = tree.get_node("root")
    root_children_id = root.successors(tree.identifier)
    total_n = 0

    for root_child_id in root_children_id:
        root_child = tree.get_node(root_child_id)
        root_child_n = root_child.data.visit_count
        root_child_move = root_child.data.move
        piece, col, row, rotation = root_child_move
        policy_index = policy_piece_to_index[piece][rotation]
        probability_matrix[policy_index][row][col + 2] = root_child_n

        total_n += root_child_n
    
    probability_matrix /= total_n
    return probability_matrix.tolist()


def simplify_grid(grid):
    # Replaces minos in a grid with 1s.
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != 0:
                grid[row][col] = 1
    return grid

# Methods for getting game data
# All of them orient the info in the perspective of the active player
def get_grids(game):
    grids = [[], []]

    for i, player in enumerate(game.players):
        grids[i] = [x[:] for x in player.board.grid] # Copy
        simplify_grid(grids[i])
    
    if game.turn == 1: grids = grids[::-1]
    
    return grids
    
def get_pieces(game):
    piece_table = np.zeros((2, 2 + PREVIEWS, len(MINOS)), dtype=int)
    for i, player in enumerate(game.players):
        if player.piece != None: # Active piece: 0
            piece_table[i][0][MINOS.index(player.piece.type)] = 1
        if player.held_piece != None: # Held piece: 1
            piece_table[i][1][MINOS.index(player.held_piece)] = 1
        # Limit previews
        for j in range(min(len(player.queue.pieces), 5)): # Queue pieces: 2-6
            piece_table[i][j + 2][MINOS.index(player.queue.pieces[j])] = 1

    if game.turn == 1: piece_table = piece_table[::-1]

    return piece_table

def get_b2b(game):
    b2b = [player.stats.b2b for player in game.players]
    if game.turn == 1: b2b = b2b[::-1]
    return b2b

def get_combo(game):
    combo = [player.stats.combo for player in game.players]
    if game.turn == 1: combo = combo[::-1]
    return combo

def get_lines_cleared(game):
    lines_cleared = [player.stats.lines_cleared for player in game.players]
    if game.turn == 1: lines_cleared = lines_cleared[::-1]
    return lines_cleared

def get_lines_sent(game):
    lines_sent = [player.stats.lines_sent for player in game.players]
    if game.turn == 1: lines_sent = lines_sent[::-1]
    return lines_sent

def game_to_X(game):
    # Returns game information for the network.
    # Orient all info in perspective to the current player
    grids = get_grids(game)
    pieces = get_pieces(game)
    b2b = get_b2b(game)
    combo = get_combo(game)
    lines_cleared = get_lines_cleared(game)
    lines_sent = get_lines_sent(game)
    color = game.players[game.turn].color
    pieces_placed = game.players[game.turn].stats.pieces

    return grids[0], pieces[0], b2b[0], combo[0], lines_cleared[0], lines_sent[0], grids[1], pieces[1], b2b[1], combo[1], lines_cleared[1], lines_sent[1], color, pieces_placed

def reflect_grid(grid):
    # Return a grid flipped horizontally
    return [row[::-1] for row in grid]

def reflect_pieces(piece_table):
    # Return a queue with pieces swapped with their mirror
    # ZLOSIJT
    # O, I, and T are the same as their mirror
    # Z -> S: 0 -> 3
    # S -> Z: 3 -> 0
    # L -> J: 1 -> 5
    # J -> L: 5 -> 1
    piece_swap_dict = {
        0: 3,
        3: 0,
        1: 5,
        5: 1
    }

    reflected_piece_table = np.zeros((2 + PREVIEWS, len(MINOS)), dtype=int)

    for i, piece_row in enumerate(piece_table):
        if 1 in piece_row:
            piece_index = piece_row.tolist().index(1) # Index of the piece in ZLOSIJT
            if piece_index in piece_swap_dict:
                # Swap piece if swappable
                piece_index = piece_swap_dict[piece_index]
            reflected_piece_table[i][piece_index] = 1
    
    return reflected_piece_table

def reflect_policy_OLD(policy_matrix, active_piece_size, hold_piece_size):
    reflected_policy_matrix = np.zeros((2, ROWS, COLS + 1, 4))
    rotation_dict = {
        1: 3,
        3: 1
    }
    for hold in range(2):
        piece_size = (active_piece_size if hold == 0 else hold_piece_size)
        for rotation in range(4):
            for col in range(COLS + 1):
                for row in range(ROWS):
                    value = policy_matrix[hold][row][col][rotation]
                    if value > 0:
                        new_rotation = rotation
                        if rotation in rotation_dict:
                            new_rotation = rotation_dict[rotation]

                        new_col = col

                        # Remove buffer
                        new_col += -2

                        # Flip column
                        new_col = 10 - new_col - piece_size # 9 - col - piece_size + 1
                        
                        # Add back buffer
                        new_col += 2

                        reflected_policy_matrix[hold][row][new_col][new_rotation] = value
    
    return reflected_policy_matrix.tolist()

def reflect_policy(policy_matrix):
    reflected_policy_matrix = np.zeros(POLICY_SHAPE)
    rotation_dict = {
        1: 3,
        3: 1
    }

    piece_swap_dict = {
        "Z": "S",
        "S": "Z",
        "L": "J",
        "J": "L"
    }

    for policy_index in range(POLICY_SHAPE[0]):
        piece, rotation = policy_index_to_piece[policy_index]

        # Save the piece size
        piece_size = len(piece_dict[piece])

        # Swap pieces that aren't the same as their mirrors
        new_piece = piece
        if new_piece in piece_swap_dict:
            new_piece = piece_swap_dict[new_piece]

        # Swap rotations that aren't the same as their mirrors
        new_rotation = rotation
        if new_rotation in rotation_dict:
            new_rotation = rotation_dict[new_rotation]
        
        # If the new rotation is a redunant shape, adjust where the piece goes
        post_col_adjustment = 0
        if new_piece in ["Z", "S", "I"]:
            # If the new rotation is 3, it needs to be shifted back after being flipped
            if new_rotation == 3:
                post_col_adjustment = -1
                new_rotation -= 2

        new_policy_index = policy_piece_to_index[new_piece][new_rotation]
        for col in range(POLICY_SHAPE[2]):
            for row in range(POLICY_SHAPE[1]):
                value = policy_matrix[policy_index][row][col]
                if value > 0:
                    new_row = row
                    new_col = col

                    # Remove buffer
                    new_col += -2

                    # Flip column
                    new_col = 10 - new_col - piece_size # 9 - col - piece_size + 1
                    
                    # Add back buffer
                    new_col += 2

                    # Add column adjustment if needed
                    new_col += post_col_adjustment

                    reflected_policy_matrix[new_policy_index][new_row][new_col] = value
    
    return reflected_policy_matrix.tolist()

def get_piece_sizes(player):
    active_piece_len = 0
    if player.piece != None:
        active_piece_len = len(piece_dict[player.piece.type])

    hold_piece_len = 0
    player_copy = player.copy()
    player_copy.hold_piece()

    if player_copy.piece != None:
        hold_piece_len = len(piece_dict[player_copy.piece.type])

    
    return active_piece_len, hold_piece_len

def play_game(network, NUMBER, show_game=False):
    # AI plays one game against itself
    # Returns the game data
    if show_game == True:
        screen = pygame.display.set_mode( (WIDTH, HEIGHT))
        pygame.display.set_caption(f'Training game {NUMBER}')

        for event in pygame.event.get():
            pass

    game = Game()
    game.setup()

    # Initialize data storage
    # Each player's move data will be stored in their respective list
    game_data = [[], []]

    while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
        # with cProfile.Profile() as pr:
        move, tree = MCTS(game, network, add_noise=True)
        search_matrix = search_statistics(tree) # Moves that the network looked at
        
        # Piece sizes are needed to know where a reflected piece ends up
        reflected_search_matrix = reflect_policy(search_matrix)

        # Get data
        move_data = [*game_to_X(game)] 

        # Reflect each player for a total of 2 * 2 = 4 times more data
        # When the other player is reflected, it shouldn't impact the piece of the active player
        # When the active player is reflected, need to alter search stats
        for active_player_idx in range(2): # 0: not reflected, 1: reflected
            for other_player_idx in range(2):
                # Copy move data
                copied_data = []
                for feature in move_data:
                    if type(feature) == np.array:
                        copied_data.append(feature.copy())
                    elif type(feature) == list:
                        copied_data.append([x[:] for x in feature])
                    else:
                        copied_data.append(feature)

                # Flip boards and pieces
                if active_player_idx == 1:
                    copied_data[0] = reflect_grid(copied_data[0])
                    copied_data[1] = reflect_pieces(copied_data[1])
                
                if other_player_idx == 1:
                    copied_data[6] = reflect_grid(copied_data[6])
                    copied_data[7] = reflect_pieces(copied_data[7])

                # Convert np arrays to regular lists
                for i in range(len(copied_data)):
                    if isinstance(copied_data[i], np.ndarray):
                        copied_data[i] = copied_data[i].tolist()

                # Flip search matrix
                if active_player_idx == 0:
                    copied_data.append(search_matrix)
                else:
                    copied_data.append(reflected_search_matrix)

                game_data[game.turn].append(copied_data)

        # move_data = [*game_to_X(game)]
        # # Convert to regular lists
        # for i in range(len(move_data)):
        #     if isinstance(move_data[i], np.ndarray):
        #         move_data[i] = move_data[i].tolist()
        
        # move_data.append(search_matrix)
        # game_data[game.turn].append(move_data)

        game.make_move(move)

        if show_game == True:
            game.show(screen)
            pygame.display.update()

        # stats = pstats.Stats(pr)
        # stats.sort_stats(pstats.SortKey.TIME)
        # stats.print_stats(20)

    # After game ends update value
    winner = game.winner
    for i in range(len(game_data)):
        if winner == -1: # Draw
            value = 0.5
        # Set values to 1 and -1
        elif winner == i:
            value = 1
        else:
            value = 0
        # Insert value before policy
        for j in range(len(game_data[i])):
            game_data[i][j].insert(-1, value)

    # Reformat data
    data = game_data[0]
    data.extend(game_data[1])

    return data

def make_training_set(network, model_version_to_load, num_games, show_game=False):
    # Creates a dataset of several AI games.
    series_data = []
    for i in range(1, num_games + 1):
        data = play_game(network, i, show_game=show_game)
        series_data.extend(data)

    json_data = json.dumps(series_data)

    # Increment set counter
    next_set = highest_data_ver(model_version_to_load) + 1

    with open(f"{directory_path}/{CURRENT_VERSION}.{model_version_to_load}.{next_set}.txt", 'w') as out_file:
        out_file.write(json_data)

def load_data(model_version_to_load, last_n_sets):
    data = []
    # Load data from the past n games
    # Find highest set number
    max_set = highest_data_ver(model_version_to_load)

    # Get n games
    for filename in get_filenames('.txt'):
        version, model_version, data_number = split_data_filename(filename)
        if (version == CURRENT_VERSION
            and model_version == model_version_to_load 
            and data_number > max_set - last_n_sets):
            # Load data
            set = json.load(open(f"{directory_path}/{filename}", 'r'))
            data.extend(set)
    return data

def battle_networks(NN_1, NN_2, threshold, network_1_title='Network 1', network_2_title='Network 2', show_game=False):
    # Battle two AI's with different networks.
    # Returns true if NN_1 wins, otherwise returns false
    wins = np.zeros((2), dtype=int)
    for i in range(BATTLE_GAMES):
        if show_game == True:
            screen = pygame.display.set_mode( (WIDTH, HEIGHT))
            pygame.display.set_caption(f'{network_1_title} | {wins[0]} vs {wins[1]} | {network_2_title}')

        game = Game()
        game.setup()
        while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
            if game.turn == 0:
                move, _ = MCTS(game, NN_1)    
            elif game.turn == 1:
                move, _ = MCTS(game, NN_2)
            game.make_move(move)

            if show_game == True:
                game.show(screen)
                pygame.display.update()

        winner = game.winner
        if winner == -1:
            wins += 0.5
        else: wins[winner] += 1

        # End early if either player reaches the cutoff
        if wins[0] >= threshold * BATTLE_GAMES:
            print(*wins)
            return True
        elif wins[1] > (1 - threshold) * BATTLE_GAMES:
            print(*wins)
            return False

    # If neither side eaches a cutoff (which shouldn't happen) return false
    return False

def battle_networks_win_loss(NN_1, NN_2, network_1_title='Network 1', network_2_title='Network 2', show_game=False):
    # Battle two AI's with different networks, and returns the wins and losses for each network
    wins = np.zeros((2), dtype=int)
    for i in range(BATTLE_GAMES):
        if show_game == True:
            screen = pygame.display.set_mode( (WIDTH, HEIGHT))
            pygame.display.set_caption(f'{network_1_title} | {wins[0]} vs {wins[1]} | {network_2_title}')

            for event in pygame.event.get():
                pass

        game = Game()
        game.setup()
        while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
            if game.turn == 0:
                move, _ = MCTS(game, NN_1)    
            elif game.turn == 1:
                move, _ = MCTS(game, NN_2)
            game.make_move(move)

            if show_game == True:
                game.show(screen)
                pygame.display.update()

        winner = game.winner
        if winner == -1:
            wins += 0.5
        else: wins[winner] += 1

    print(network_1_title, wins, network_2_title)
    return wins

def self_play_loop(show_games=False):
    if show_games == True:
        pygame.init()
    # Given a network, generates training data, trains it, and checks if it improved.
    best_network = load_best_network()
    iter = 0

    while True:
        iter += 1
        # challenger_network = keras.models.clone_model(best_network)
        # Cloned models are worse than loading models
        # ARGHHHH
        # The challenger network will be trained and then battled against the prior network
        challenger_network = load_best_network()
        best_network_version = highest_model_ver()

        # Play a training set and train the network on past sets.
        for i in range(TRAINING_LOOPS):
            # Make data file
            make_training_set(challenger_network, best_network_version, TRAINING_GAMES, show_game=show_games)
            print("Finished set")

            # Load data generated by the best current network
            data = load_data(best_network_version, SETS_TO_TRAIN_WITH)
        
            # Use 1/10th of the available data
            # data_batch, _ = train_test_split(data, train_size=0.1)
            print(len(data))#, len(data_batch))

            # Train challenger network
            train_network(challenger_network, data)
        print("Finished loop")

        # If new network is improved, save it and make it the default
        # Otherwise, repeat
        if battle_networks(challenger_network, best_network, 0.55, show_game=show_games):
            # Challenger network becomes next highest version
            next_ver = highest_model_ver() + 1
            challenger_network.save(f"{directory_path}/{CURRENT_VERSION}.{next_ver}.keras")
            # The new network becomes the network to beat
            best_network = challenger_network

def load_best_network():
    max_ver = highest_model_ver()

    path = f"{directory_path}/{CURRENT_VERSION}.{max_ver}.keras"

    print(path)

    return tf.keras.models.load_model(path)

def highest_model_ver():
    max = -1

    for filename in get_filenames('.keras'):
        version, model_version = split_model_filename(filename)
        if (version == CURRENT_VERSION
            and model_version > max):
            max = model_version

    return max

def highest_data_ver(best_network_version):
    max = -1

    for filename in get_filenames('.txt'):
        version, model_version, data_number = split_data_filename(filename)
        if (version == CURRENT_VERSION 
            and model_version == best_network_version
            and data_number > max):
            max = data_number

    return max

def split_data_filename(filename):
    version_part_1, filename = filename.split('.', 1)
    version_part_2, filename = filename.split('.', 1)
    model_version, cut_filename = filename.split('.', 1)
    number = cut_filename.split('.', 1)[0]

    return float(f"{version_part_1}.{version_part_2}"), int(model_version), int(number)

def split_model_filename(filename):
    version_part_1, filename = filename.split('.', 1)
    version_part_2, filename = filename.split('.', 1)
    model_version = filename.split('.', 1)[0]

    return float(f"{version_part_1}.{version_part_2}"), int(model_version)

def get_filenames(extension):
    filenames = []

    for filename in os.listdir(f'{directory_path}'):
        if filename.startswith(str(CURRENT_VERSION)) and filename.endswith(extension):
            filenames.append(filename)
    
    return filenames

# Debug Functions

if __name__ == "__main__":
    def visualize_piece_placements(game, moves):
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Tetris')
        # Need to iterate through pygame events to initialize screen
        for event in pygame.event.get():
            pass

        for policy, move in moves:
            game_copy = game.copy()
            game_copy.make_move(move)

            game_copy.show(screen)
            pygame.display.update()

            time.sleep(0.3)

    game = Game()
    game.setup()

    # Place a piece to make it more interesting
    for i in range(10):
        piece = game.players[game.turn].piece
        game.make_move((piece.type, piece.x_0, game.players[game.turn].ghost_y, 0))

    move_matrix = get_move_matrix(game.players[game.turn])
    moves = get_move_list(move_matrix, np.ones(shape=POLICY_SHAPE))

    visualize_piece_placements(game, moves)

    # Now, get the reflected moves
    player = game.players[game.turn]
    # Reflect board
    player.board.grid = reflect_grid(player.board.grid)

    # Reflect pieces
    piece_table = get_pieces(game)[0]
    reflected_piece_table = reflect_pieces(piece_table)
    for idx, piece_row in enumerate(reflected_piece_table):
        if idx == 0:
            player.piece.type = MINOS[piece_row.tolist().index(1)]
        elif idx == 1:
            if player.held_piece != None: # Active piece: 0
                player.held_piece.type = MINOS[piece_row.tolist().index(1)]
        else:
            player.queue.pieces[idx - 2] = MINOS[piece_row.tolist().index(1)]
    
    # Reflect the policy and see if it matches
    reflected_move_matrix = reflect_policy(move_matrix)
    reflected_moves = get_move_list(reflected_move_matrix, np.ones(shape=POLICY_SHAPE))

    visualize_piece_placements(game, reflected_moves)

# pygame.init()
# loaded_model = load_best_network()
# clone_model = keras.models.clone_model(loaded_model)
# print(battle_networks_win_loss(loaded_model, clone_model, 
#                                network_1_title="Loaded Netowrk",
#                                network_2_title="Cloned network",
#                                show_game=True))