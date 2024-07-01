from const import *
from game import Game

from collections import deque
import gc
#import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import time
import treelib
import ujson

# # Prior to importing tensorflow, disable debug logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow import keras
import tensorflow as tf
from tensorflow.python.ops import math_ops
from sklearn.model_selection import train_test_split

# Reduce tensorflow text
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

import cProfile
import pstats

# For naming data and models
CURRENT_VERSION = 40.3

# Where data and models are saved
directory_path = '/Users/matthewlee/Documents/Code/Tetris Game/Storage'

# Areas of optimization:
# - Finding piece locations/generating move matrix (O pieces no rotation, use faster algo)
# - Load data faster

# AI todo:
# - Battle networks (Size/Speed) (L2, CNNs, adjust number of weights/layers, Dropout)
# - Battle parameters (CPuct, Dirichlet noise, loss weights)
# - Encoding garbage into the neural network
# - Shuffle loading data
# - Change how data is savaed and loaded
# - Read/implement paper on accelerating self play
# - Investigate hold issue (maybe search depth?)

total_branch = 0
number_branch = 0

class Config():
    def __init__(self, 
                 l1_neurons=256, 
                 l2_neurons=32, 
                 learning_rate=0.001, 
                 loss_weights=[1, 1], 
                 epochs=1, 
                 MAX_ITER=160, # 1600
                 DIRICHLET_ALPHA=0.01, 
                 DIRICHLET_EXPLORATION=0.25, 
                 CPUCT=1):
        
        self.l1_neurons = l1_neurons
        self.l2_neurons = l2_neurons
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.epochs = epochs
        self.MAX_ITER = MAX_ITER
        self.DIRICHLET_ALPHA = DIRICHLET_ALPHA
        self.DIRICHLET_EXPLORATION = DIRICHLET_EXPLORATION
        self.CPUCT = CPUCT

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

        self.visit_count = 0
        # self.visit_count = 1
        self.value_sum = 0
        self.value_avg = 0
        self.policy = 0

def MCTS(config, game, network, add_noise=False, move_algorithm='faster-but-loss'):
    global total_branch, number_branch
    # Picks a move for the AI to make 
    # Initialize the search tree
    tree = treelib.Tree()
    game_copy = game.copy()

    # Restrict previews
    for player in game_copy.players:
        while len(player.queue.pieces) > PREVIEWS:
            player.queue.pieces.pop(-1)

    # Create the initial node
    initial_state = NodeState(game=game_copy, move=None)

    tree.create_node(identifier="root", data=initial_state)

    MAX_DEPTH = 0
    iter = 0

    while iter < config.MAX_ITER:
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
            
            # Info to help me debug
            Qs = []
            Us = []

            for child_id in child_ids:
                
                total_branch += 1

                # For each child calculate a score
                # Polynomial upper confidence trees (PUCT)
                child_data = tree.get_node(child_id).data
                child_score = child_data.value_avg + config.CPUCT * child_data.policy*math.sqrt(parent_visits)/(1+child_data.visit_count)

                Qs.append(child_data.value_avg)
                Us.append(config.CPUCT* child_data.policy*math.sqrt(parent_visits)/(1+child_data.visit_count))

                if child_score >= max_child_score:
                    max_child_score = child_score
                    max_child_id = child_id
            
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
            value, policy = evaluate_from_tflite(node_state.game, network)
            # value, policy = random_evaluate()

            if node_state.game.no_move == False:
                move_matrix = get_move_matrix(node_state.game.players[node_state.game.turn], algo=move_algorithm)
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
            if winner == node_state.game.turn: # If active player wins
                value = 1
            elif winner == 1 - node_state.game.turn: # If opponent wins
                value = 0
            else: # Draw
                value = 0.5

        # If root node and in self play, add exploration noise to children
        if (add_noise == True and node.is_root()):
            child_ids = node.successors(tree.identifier)
            number_of_children = len(child_ids)
            noise_distribution = np.random.gamma(config.DIRICHLET_ALPHA, 1, number_of_children)

            pre_noise_policy = []
            post_noise_policy = []

            for child_id, noise in zip(child_ids, noise_distribution):
                child_data = tree.get_node(child_id).data
                pre_noise_policy.append(child_data.policy)

                child_data.policy = child_data.policy * (1 - config.DIRICHLET_EXPLORATION) + noise * config.DIRICHLET_EXPLORATION
                post_noise_policy.append(child_data.policy)
            
            # fig, axs = plt.subplots(2)
            # fig.suptitle('Policy before and after dirichlet noise')
            # axs[0].plot(pre_noise_policy)
            # axs[1].plot(post_noise_policy)
            # plt.savefig(f"{directory_path}/policy_3_7_1")
            # print("saved")

        # When you make a make a move and evaluate it, the turn flips so
        # the evaluation is from the perspective of the other player, 
        # thus you have to flip the value
        value = 1-value

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

    # print(MAX_DEPTH, total_branch//number_branch)

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
        if root_child_n >= max_n: # It's possible n is 0 if there are no possible moves
            max_n = root_child_n
            max_id = root_child.identifier

    # fig, axs = plt.subplots(2)
    # fig.suptitle('N Compared to policy')
    # axs[0].plot(post_noise_policy)
    # axs[1].plot(root_child_n_list)
    # plt.savefig(f"{directory_path}/root_n_{MAX_ITER}_depth_3_4_2")
    # print("saved")

    data = tree.get_node(max_id).data
    move = data.move

    return move, tree

def get_move_matrix(player, algo=None):
    # Returns a list of all possible moves that a player can make
    # Very Important: With my coordinate system, pieces can be placed with negative x values
    # To avoid negative indexing, the list of moves is shifted by 2
    # It encodes moves from -2 to 8 as indices 0 to 10
    #
    # Algorithm names:
        # 'brute-force'
        # 'faster-but-loss'
    # Possible locations needs to be larger to acount for blocks with negative x
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

    def check_add_to_sets(x, y, o, check_placement=True):
        # Checks if move has been looked at already
        if (x, y, o) not in check_set:
            next_location_queue.append((x, y, o))
            check_set.add((x, y, o))

            # Check if it can be placed
            if check_placement:
                coords = [[col + x, row + y + 1] for col, row in mino_coords_dict[piece.type][o]]
                if sim_player.collision(coords):
                    place_location_queue.append((x, y, o))

    def algo_1(check_rotations):
        # My initial algorithm
        # Will always find every piece placement
        # Takes a long time
        while len(next_location_queue) > 0:
            location = next_location_queue.popleft()

            piece.x_0, piece.y_0, piece.rotation = location

            piece.coordinates = piece.get_self_coords

            # Check left, right moves
            for x_offset in [1, -1]:
                if sim_player.can_move(piece, x_offset=x_offset): # Check this first to avoid index errors
                    x = piece.x_0 + x_offset
                    y = piece.y_0
                    o = piece.rotation

                    check_add_to_sets(x, y, o)
            
            # Check full softdrop
            ghost_y = sim_player.ghost_y

            if ghost_y != sim_player.piece.y_0:
                x = piece.x_0
                o = piece.rotation

                check_add_to_sets(x, ghost_y, o)

            # Check left, right, and down moves
            for move in [[1, 0], [-1, 0], [0, 1]]:
                if sim_player.can_move(piece, x_offset=move[0], y_offset=move[1]): # Check this first to avoid index errors
                    x = piece.x_0 + move[0]
                    y = piece.y_0 + move[1]
                    o = piece.rotation

                    check_add_to_sets(x, y, o)

            if check_rotations:
                # Check rotations 1, 2, and 3
                for i in range(1, 4):
                    sim_player.try_wallkick(i)

                    x = piece.x_0
                    y = piece.y_0
                    o = piece.rotation

                    if y >= 0: # Avoid negative indexing
                        check_add_to_sets(x, y, o)

                    # Reset piece locations
                    if i != 3: # Don't need to reset on last rotation
                        piece.x_0, piece.y_0, piece.rotation = location

    def algo_2(check_rotations):
        # Faster algorithm
        # Requires spawn height to be above or equal to max height
        #
        # For each piece:
        # If the maxheight is below or the same as the piece spawn height:
            # Phase 1:
                # Rotate the piece for each rotation
            # Phase 2:
                # Move each rotation to each different column
            # Phase 3:
                # Move each of those all the way down, marking each spot on the way down
            # Phase 4:
                # Do the normal move gen
        # If the maxheight is above the piece spawn height:
            # Do normal move gen

        phase_2_queue = deque()

        # Phase 1
        location = next_location_queue.popleft()
        piece.x_0, piece.y_0, piece.rotation = location
        piece.coordinates = piece.get_self_coords

        phase_2_queue.append((piece.x_0, piece.y_0, piece.rotation))

        if check_rotations:
            for i in range(1, 4):
                sim_player.try_wallkick(i)

                x = piece.x_0
                y = piece.y_0
                o = piece.rotation

                phase_2_queue.append((x, y, o))
                check_set.add((x, y, o))

                if i != 3:
                    piece.x_0, piece.y_0, piece.rotation = location
        
        phase_3_queue = deque()

        # Phase 2
        while len(phase_2_queue) > 0:
            location = phase_2_queue.popleft()
            phase_3_queue.append(location)

            for x_dir in [-1, 1]:
                piece.x_0, piece.y_0, piece.rotation = location
                piece.coordinates = piece.get_self_coords

                while sim_player.can_move(piece, x_offset=x_dir):
                    x = piece.x_0 + x_dir
                    y = piece.y_0
                    o = piece.rotation

                    piece.x_0 = x
                    piece.coordinates = piece.get_self_coords

                    phase_3_queue.append((x, y, o))
                    check_set.add((x, y, o))

        # Phase 3
        while len(phase_3_queue) > 0:
            location = phase_3_queue.popleft()
            piece.x_0, piece.y_0, piece.rotation = location
            piece.coordinates = piece.get_self_coords

            x = piece.x_0
            y = piece.y_0
            o = piece.rotation

            while sim_player.can_move(piece, y_offset=1):
                x = piece.x_0
                y = piece.y_0 + 1
                o = piece.rotation

                piece.y_0 = y
                piece.coordinates = piece.get_self_coords

                # Add these to check set but not queue until they hit the bottom
                check_set.add((x, y, o))

            # Check these using the normal algorithm
            next_location_queue.append((x, y, o))

            # Piece must be placeable by definition
            place_location_queue.append((x, y, o))

        # Phase 4 is the regular algorithm
        algo_1(check_rotations)


    # Other ideas:
    # Calculate all areas that a piece could fit, subtract the moves that can be found
    # by harddropping, and then backtrack to find unfound moves
        # Issue of backtracking kicks

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
            # Queue for looking through piece placements
            next_location_queue = deque()
            place_location_queue = []
            check_set = set()

            # Start the piece at the highest point it can be placed
            highest_row = get_highest_row(sim_player.board.grid)
            starting_row = max(highest_row - len(piece_dict[piece.type]), ROWS - SPAWN_ROW)
            piece.y_0 = starting_row

            check_add_to_sets(piece.x_0, piece.y_0, piece.rotation)

            # O piece can't rotate
            check_rotations = True
            if piece.type == "O":
                check_rotations = False

            # Search through the queue
            if algo == 'brute-force':
                algo_1(check_rotations)
            elif algo == 'faster-but-loss':
                algo_2(check_rotations)
            else:
                raise Exception("Invalid algorithm type")

            for x, y, o in place_location_queue:
                rotation_index = o % len(policy_pieces[piece.type])
                policy_index = policy_piece_to_index[piece.type][rotation_index]

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

                new_policy[policy_index][new_row][new_col + 2] = 1 # Account for buffer        
        
    return new_policy

def get_move_list(move_matrix, policy_matrix):
    # # Returns list of possible moves with their policy
    # # Removes buffer
    mask = np.multiply(move_matrix, policy_matrix)

    move_list = np.argwhere(mask != 0)

    # Formats moves from (policy index, row, col) to (value, (policy index, col - 2, row))
    move_list = [(mask[move[0]][move[1]][move[2]], (move[0], move[2] - 2, move[1])) for move in move_list]

    return move_list

# Using deepcopy:                          100 iter in 36.911 s
# Using copy functions in classes:         100 iter in 1.658 s
# Many small changes:                      100 iter in 1.233 s
# MCTS uses game instead of player:        100 iter in 1.577 s
# Added large NN but optimized MCTS:       100 iter in 7.939 s
#   Without NN:                            100 iter in 0.882 s
#   Changed collision and added coords:    100 iter in 0.713 s
# Use Model(X) instead of .predict:        100 iter in 3.506 s
# Use Model.predict_on_batch(X):           100 iter in 1.788 s
# Use TFlite model + argwhere and full sd: 100 iter in 0.748 s

##### Neural Network #####
# 
# Player orientation: Active player, other player
# y:
#   Policy: (19 x 25 x 11) = 5225 (Hold x Rows x Columns x Rotations)
#   Value: (1)

def create_input_layers():
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

    inputs = []
    active_features = []
    active_grid = None
    opponent_features = []
    opponent_grid = None
    
    non_player_features = []

    for i, shape in enumerate(shapes):
        # Add input
        input = keras.Input(shape=shape, name=f"{i}")
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
                opponent_grid = input
            else:
                opponent_features.append(keras.layers.Flatten()(input))
        # Other features
        else:
            non_player_features.append(keras.layers.Flatten()(input))
    
    return inputs, active_grid, active_features, opponent_grid, opponent_features, non_player_features

def gen_fishlike_nn(config: Config) -> keras.Model:
    # The network is designed to be fast and compatable with tflite
    # Drawing inspiration from stockfish's NNUE architecture

    def ValueHead():
        # Returns value; found at the end of the network
        def inside(x):
            x = keras.layers.Dense(1, activation="sigmoid")(x)

            return x
        return inside

    def PolicyHead():
        # Returns policy list; found at the end of the network
        def inside(x):
            # Generate probability distribution
            x = keras.layers.Dense(POLICY_SIZE, activation="softmax")(x)

            return x
        return inside

    inputs, a_grid, a_features, o_grid, o_features, non_player_features = create_input_layers()

    flat_a_grid = keras.layers.Flatten()(a_grid)
    a_features = keras.layers.Concatenate()([flat_a_grid, *a_features])
    side_a = keras.layers.Dense(config.l1_neurons)(a_features)
    side_a = keras.layers.Activation('relu')(side_a)

    flat_o_grid = keras.layers.Flatten()(o_grid)
    o_features = keras.layers.Concatenate()([flat_o_grid, *o_features])
    side_o = keras.layers.Dense(config.l1_neurons)(o_features)
    side_o = keras.layers.Activation('relu')(side_o)

    combined = keras.layers.Concatenate()([side_a, side_o, *non_player_features])

    combined = keras.layers.Dense(config.l2_neurons)(combined)
    combined = keras.layers.Activation('relu')(combined)

    combined = keras.layers.Dense(config.l2_neurons)(combined)
    combined = keras.layers.Activation('relu')(combined)

    value_output = ValueHead()(combined)
    policy_output = PolicyHead()(combined)

    model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])

    return model

def gen_alphastack_nn(config: Config) -> keras.Model:
    # The network is designed to be fast and compatable with tflite
    # Drawing inspiration from stockfish's NNUE architecture

    filters = 16
    dropout = 0.4

    def ValueHead():
        # Returns value; found at the end of the network
        def inside(x):
            x = keras.layers.Dense(1, activation="sigmoid")(x)

            return x
        return inside

    def PolicyHead():
        # Returns policy list; found at the end of the network
        def inside(x):
            # Generate probability distribution
            x = keras.layers.Dense(POLICY_SIZE, activation="softmax")(x)

            return x
        return inside

    def ResidualLayer():
        # Uses skip conections
        def inside(x):
            y = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Activation('relu')(y)
            y = keras.layers.Conv2D(filters, (3, 3), padding="same")(y)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Activation('relu')(y)
            z = keras.layers.Add()([x, y]) # Skip connection

            z = keras.layers.Dropout(dropout)(z)

            return z
        return inside

    inputs, a_grid, a_features, o_grid, o_features, non_player_features = create_input_layers()

    stacked_grids = keras.layers.Concatenate(axis=-1)([a_grid, o_grid])
    # stacked_grids = a_grid

    # Apply a convolution
    stacked_grids = keras.layers.Conv2D(filters, (3, 3), padding="same")(stacked_grids)
    stacked_grids = keras.layers.BatchNormalization()(stacked_grids)
    stacked_grids = keras.layers.Activation('relu')(stacked_grids)
    stacked_grids = keras.layers.Dropout(dropout)(stacked_grids)

    # 10 residual layers
    for _ in range(10):
        stacked_grids = ResidualLayer()(stacked_grids)
    
    # 1x1 Kernel
    x = keras.layers.Conv2D(1, (1, 1))(stacked_grids)
    x = keras.layers.Flatten()(x)

    # Combine with other features
    x = keras.layers.Concatenate()([x, *a_features, *o_features, *non_player_features])

    value_output = ValueHead()(x)
    policy_output = PolicyHead()(x)

    model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])

    return model

def gen_alphasplit_nn(config: Config) -> keras.Model:
    # The network is designed to be fast and compatable with tflite
    # Drawing inspiration from stockfish's NNUE architecture

    filters = 16
    dropout = 0.4

    def ValueHead():
        # Returns value; found at the end of the network
        def inside(x):
            x = keras.layers.Dense(1, activation="sigmoid")(x)

            return x
        return inside

    def PolicyHead():
        # Returns policy list; found at the end of the network
        def inside(x):
            # Generate probability distribution
            x = keras.layers.Dense(POLICY_SIZE, activation="softmax")(x)

            return x
        return inside

    def ResidualLayer():
        # Uses skip conections
        def inside(x):
            y = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Activation('relu')(y)
            y = keras.layers.Conv2D(filters, (3, 3), padding="same")(y)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Activation('relu')(y)
            z = keras.layers.Add()([x, y]) # Skip connection

            z = keras.layers.Dropout(dropout)(z)

            return z
        return inside

    inputs, a_grid, a_features, o_grid, o_features, non_player_features = create_input_layers()

    # Apply a convolution
    a_grid = keras.layers.Conv2D(filters, (3, 3), padding="same")(a_grid)
    a_grid = keras.layers.BatchNormalization()(a_grid)
    a_grid = keras.layers.Activation('relu')(a_grid)
    a_grid = keras.layers.Dropout(dropout)(a_grid)

    o_grid = keras.layers.Conv2D(filters, (3, 3), padding="same")(o_grid)
    o_grid = keras.layers.BatchNormalization()(o_grid)
    o_grid = keras.layers.Activation('relu')(o_grid)
    o_grid = keras.layers.Dropout(dropout)(o_grid)

    # 10 residual layers
    for _ in range(10):
        a_grid = ResidualLayer()(a_grid)
        o_grid = ResidualLayer()(o_grid)
    
    # 1x1 Kernel
    a_grid = keras.layers.Conv2D(1, (1, 1))(a_grid)
    a_grid = keras.layers.Flatten()(a_grid)

    o_grid = keras.layers.Conv2D(1, (1, 1))(o_grid)
    o_grid = keras.layers.Flatten()(o_grid)

    # Shrink opponent grid info
    o_grid = keras.layers.Dense(16)(o_grid)

    # Combine with other features
    x = keras.layers.Concatenate()([a_grid, o_grid, *a_features, *o_features, *non_player_features])

    value_output = ValueHead()(x)
    policy_output = PolicyHead()(x)

    model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])

    return model

def gen_alphasame_nn(config: Config) -> keras.Model:
    # The network is designed to be fast and compatable with tflite
    # Drawing inspiration from stockfish's NNUE architecture

    filters = 16
    dropout = 0.4

    def ValueHead():
        # Returns value; found at the end of the network
        def inside(x):
            x = keras.layers.Dense(1, activation="sigmoid")(x)

            return x
        return inside

    def PolicyHead():
        # Returns policy list; found at the end of the network
        def inside(x):
            # Generate probability distribution
            x = keras.layers.Dense(POLICY_SIZE, activation="softmax")(x)

            return x
        return inside

    def ResidualLayer():
        # Uses skip conections
        def inside(in_1, in_2):
            conv_1 = keras.layers.Conv2D(filters, (3, 3), padding="same")
            batch_1 = keras.layers.BatchNormalization()
            relu_1 = keras.layers.Activation('relu')
            conv_2 = keras.layers.Conv2D(filters, (3, 3), padding="same")
            batch_2 = keras.layers.BatchNormalization()
            relu_2 = keras.layers.Activation('relu')

            out_1 = relu_2(batch_2(conv_2(relu_1(batch_1(conv_1(in_1))))))
            out_2 = relu_2(batch_2(conv_2(relu_1(batch_1(conv_1(in_2))))))

            out_1 = keras.layers.Add()([in_1, out_1])
            out_2 = keras.layers.Add()([in_2, out_2])

            dropout_1 = keras.layers.Dropout(dropout)

            out_1 = dropout_1(out_1)
            out_2 = dropout_1(out_2)

            return out_1, out_2
        return inside

    inputs, a_grid, a_features, o_grid, o_features, non_player_features = create_input_layers()

    # Start with a convolutional layer
    # Because each grid needs the same network, use the same layers for each side
    conv_1 = keras.layers.Conv2D(filters, (3, 3), padding="same")
    batch_1 = keras.layers.BatchNormalization()
    relu_1 = keras.layers.Activation('relu')
    dropout_1 = keras.layers.Dropout(dropout)
    
    a_grid = dropout_1(relu_1(batch_1(conv_1(a_grid))))
    o_grid = dropout_1(relu_1(batch_1(conv_1(o_grid))))

    # 10 residual layers
    for _ in range(10):
        residual_layer = ResidualLayer()
        a_grid, o_grid = residual_layer(a_grid, o_grid)
    
    # 1x1 Kernel
    kernel_1 = keras.layers.Conv2D(1, (1, 1))
    a_grid = kernel_1(a_grid)
    o_grid = kernel_1(o_grid)

    flatten_1 = keras.layers.Flatten()
    a_grid = flatten_1(a_grid)
    o_grid = flatten_1(o_grid)

    # Shrink opponent grid info
    o_grid = keras.layers.Dense(16)(o_grid)

    # Combine with other features
    x = keras.layers.Concatenate()([a_grid, o_grid, *a_features, *o_features, *non_player_features])

    value_output = ValueHead()(x)
    policy_output = PolicyHead()(x)

    model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])

    return model

def instantiate_network(config: Config, nn_generator, show_summary=True, save_network=True, plot_model=False):
    # Creates a network with random weights
    # 1: For each grid, apply the same neural network, and then use 1x1 kernel and concatenate
    # 1 -> 2: For opponent grid, apply fully connected layer
    # Concatenate active player's kernels/features with opponent's dense layer and non-player specific features
    # Apply value head and policy head 

    model = nn_generator(config)

    if plot_model == True:
        keras.utils.plot_model(model, to_file=f"{directory_path}/model_{CURRENT_VERSION}_img.png", show_shapes=True)

    # Loss is the sum of MSE of values and Cross entropy of policies
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate), loss=["mean_squared_error", "categorical_crossentropy"], loss_weights=config.loss_weights)

    if show_summary: model.summary()

    if save_network: model.save(f"{directory_path}/{CURRENT_VERSION}.0.keras")

    return model

def train_network(config, model, data):
    # Don't train over everything simultaneously
    # Keras has 2GB limit (?)
    for set in data:
        # Fit the model
        # Swap rows and columns
        features = list(map(list, zip(*set)))

        # Make features into np arrays
        for i in range(len(features)):
            features[i] = np.array(features[i])

        # Last two columns are value and policy
        policies = features.pop()
        values = features.pop()

        # Reshape policies
        policies = np.array(policies).reshape((-1, POLICY_SIZE))

        # callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience = 20)

        model.fit(x=features, y=[values, policies], batch_size=64, epochs=config.epochs, shuffle=True)

def evaluate_from_tflite(game, interpreter):
    # Use a neural network to return value and policy.
    data = game_to_X(game)
    X = []
    for feature in data:
        if type(feature) in (float, int):
            X.append(np.expand_dims(np.float32(feature), axis=(0, 1)))
        else:
            np_feature = np.expand_dims(np.float32(feature), axis=0)
            if np_feature.shape == (1, 26, 10): # Expand grids
                np_feature = np.expand_dims(np_feature, axis=-1)
            X.append(np_feature)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(X)):
        split_str = input_details[i]['name'].split(":")[0]
        
        if len(split_str) == 12: # "serving_default"
            idx = 0
        else: 
            split_str = split_str.split("_")[2]
            idx = int(split_str)
        
        interpreter.set_tensor(input_details[i]["index"], X[idx])

    interpreter.invoke()

    value = interpreter.get_tensor(output_details[1]['index'])
    policies = interpreter.get_tensor(output_details[0]['index'])
    
    # input_details = interpreter.get_input_details()
    # for i in range(len(X)):
    #     idx = int(input_details[i]['name'][16:].split(":")[0])

    # interpreter.set_tensor(interpreter.get_input_details()[i]["index"], X[idx])
    # interpreter.invoke()
    # result = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])


    # value, policies = signature_runner(inputs=X)

    # Both value and policies are returned as arrays
    policies = policies.reshape(POLICY_SHAPE)
    
    return value[0][0], policies

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

    probability_matrix = np.zeros(POLICY_SHAPE, dtype=int).tolist()

    root = tree.get_node("root")
    root_children_id = root.successors(tree.identifier)
    total_n = 0

    for root_child_id in root_children_id:
        total_n += tree.get_node(root_child_id).data.visit_count

    assert total_n != 0

    for root_child_id in root_children_id:
        root_child = tree.get_node(root_child_id)
        root_child_n = root_child.data.visit_count
        if root_child_n != 0:
            root_child_move = root_child.data.move
            policy_index, col, row = root_child_move

            # ACCOUNT FOR BUFFER
            probability_matrix[policy_index][row][col + 2] = round(root_child_n / total_n, 4)

    return probability_matrix


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

def reflect_policy(policy_matrix):
    reflected_policy_matrix = np.zeros(POLICY_SHAPE, dtype=int).tolist()
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
    
    return reflected_policy_matrix

def play_game(config, network, NUMBER, show_game=False, screen=None):
    # AI plays one game against itself
    # Returns the game data
    if show_game == True:
        if screen == None:
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
        move, tree = MCTS(config, game, network, add_noise=True) # Add training noise
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
                    if isinstance(feature, np.ndarray):
                        copied_data.append(feature.copy())
                    elif type(feature) == list:
                        copied_data.append([x[:] for x in feature])
                    else: # int or float
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

        '''move_data = [*game_to_X(game)]
        # Convert to regular lists
        for i in range(len(move_data)):
            if isinstance(move_data[i], np.ndarray):
                move_data[i] = move_data[i].tolist()
        
        move_data.append(search_matrix)
        game_data[game.turn].append(move_data)'''

        game.make_move(move)

        if show_game == True:
            game.show(screen)
            pygame.display.update()

        # stats = pstats.Stats(pr)
        # stats.sort_stats(pstats.SortKey.TIME)
        # stats.print_stats(20)

    # After game ends update value
    winner = game.winner
    for player_idx in range(len(game_data)):
        if winner == -1: # Draw
            value = 0.5
        # Set values to 1 and -1
        elif winner == player_idx:
            value = 1
        else:
            value = 0
        # Insert value before policy for each move of that player
        for move_idx in range(len(game_data[player_idx])):
            game_data[player_idx][move_idx].insert(-1, value)

    # Reformat data to stack moves into one list
    data = game_data[0]
    data.extend(game_data[1])

    return data

def make_training_set(config, network, model_version_to_load, num_games, show_game=False, screen=None):
    # Creates a dataset of several AI games.
    series_data = []
    for i in range(1, num_games + 1):
        data = play_game(config, network, i, show_game=show_game, screen=screen)
        series_data.extend(data)

    json_data = ujson.dumps(series_data)

    # Increment set counter
    next_set = highest_data_ver(model_version_to_load) + 1

    with open(f"{directory_path}/{CURRENT_VERSION}.{model_version_to_load}.{next_set}.txt", 'w') as out_file:
        out_file.write(json_data)

def load_data(model_ver=None, model_iter=None, last_n_sets=20):
    data = []
    # Load data from the past n games
    # You can input model_ver or model_iter as None to load 
    # all iterations from one version OR load all data files
    # under past last_n

    # Find highest set number
    # SHOULD ONLY BE USED IF model_ver AND model_iter are specified
    max_set = highest_data_ver(model_iter)

    sets, len_sets = 0, 0

    # Get n games
    for filename in get_filenames('.txt'):
        version, iteration, data_number = split_data_filename(filename)
        if version == model_ver or model_ver == None:
            if iteration == model_iter or model_iter == None:
                if data_number > max_set - last_n_sets:
                    # Load data
                    set = ujson.load(open(f"{directory_path}/{filename}", 'r'))
                    data.append(set)

                    sets += 1
                    len_sets += len(set)
    
    print(sets, len_sets)
    return data

def battle_networks(NN_1, config_1, NN_2, config_2, threshold, games, network_1_title='Network 1', network_2_title='Network 2', show_game=False, screen=None):
    # Battle two AI's with different networks.
    # Returns true if NN_1 wins, otherwise returns false
    wins = np.zeros((2), dtype=int)
    for i in range(games):
        if show_game == True:
            if screen == None:
                screen = pygame.display.set_mode( (WIDTH, HEIGHT))
            pygame.display.set_caption(f'{network_1_title} | {wins[0]} vs {wins[1]} | {network_2_title}')

            for event in pygame.event.get():
                pass

        game = Game()
        game.setup()
        while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
            if game.turn == 0:
                move, _ = MCTS(config_1, game, NN_1)    
            elif game.turn == 1:
                move, _ = MCTS(config_2, game, NN_2)
            game.make_move(move)

            if show_game == True:
                game.show(screen)
                pygame.display.update()

        winner = game.winner
        if winner == -1:
            wins += 0.5
        else: wins[winner] += 1

        # End early if either player reaches the cutoff
        if wins[0] >= threshold * games:
            print(*wins)
            return True
        elif wins[1] > (1 - threshold) * games:
            print(*wins)
            return False

    # If neither side eaches a cutoff (which shouldn't happen) return false
    return False

def self_play_loop(config, skip_first_set=False, show_games=False):
    if show_games == True:
        pygame.init()
        screen = pygame.display.set_mode( (WIDTH, HEIGHT))
    # Given a network, generates training data, trains it, and checks if it improved.
    best_network = load_best_model()
    best_interpreter = get_interpreter(best_network)

    del best_network

    iter = 0

    while True:
        iter += 1

        # The challenger network will be trained and then battled against the prior network
        challenger_network = load_best_model()

        best_network_version = highest_model_ver(CURRENT_VERSION)

        # Play a training set and train the network on past sets.
        for i in range(TRAINING_LOOPS):
            # Make data file
            if not skip_first_set:
                make_training_set(config, best_interpreter, best_network_version, TRAINING_GAMES, show_game=show_games, screen=screen)
                print("Finished set")

        # Load data generated by the best current network
        data = load_data(CURRENT_VERSION, best_network_version, SETS_TO_TRAIN_WITH)

        # Train challenger network
        train_network(config, challenger_network, data)

        del data

        challenger_interpreter = get_interpreter(challenger_network)
        print("Finished loop")

        # If new network is improved, save it and make it the default
        # Otherwise, repeat
        if battle_networks(challenger_interpreter, config, best_interpreter, config, 0.55, BATTLE_GAMES, show_game=show_games, screen=screen):
            # Challenger network becomes next highest version
            next_ver = highest_model_ver(CURRENT_VERSION) + 1
            challenger_network.save(f"{directory_path}/{CURRENT_VERSION}.{next_ver}.keras")

            # The new network becomes the network to beat
            best_network = challenger_network
            best_interpreter = get_interpreter(best_network)
        
        del challenger_network
        del challenger_interpreter

        gc.collect

def load_best_model():
    max_ver = highest_model_ver(CURRENT_VERSION)

    path = f"{directory_path}/{CURRENT_VERSION}.{max_ver}.keras"

    print(path)

    model = keras.models.load_model(path)

    return model

def get_interpreter(model):
    # Save a model as saved model, then load it as tflite
    path = f"{directory_path}/TEMP_MODELS/savedmodel"
    model.export(path)

    converter = tf.lite.TFLiteConverter.from_saved_model(path)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]

    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    return interpreter

def highest_model_ver(ver):
    max = -1

    for filename in get_filenames('.keras'):
        version, model_version = split_model_filename(filename)
        if (version == ver
            and model_version > max):
            max = model_version

    return max

def highest_data_ver(ver):
    max = -1

    for filename in get_filenames('.txt'):
        version, model_version, data_number = split_data_filename(filename)
        if (version == CURRENT_VERSION 
            and model_version == ver
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
        if filename.endswith(extension):
            filenames.append(filename)
    
    return filenames

# Debug Functions



if __name__ == "__main__":

    game = Game()
    game.setup()

    # Place a piece to make it more interesting
    for i in range(10):
        piece = game.players[game.turn].piece
        game.make_move((piece.type, piece.x_0, game.players[game.turn].ghost_y, 0))

    

    # Profile make moves
    with cProfile.Profile() as pr:
        for i in range(100):
            get_move_matrix(game.players[game.turn])
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(20)





    '''
    # Testing if reflecting pieces, grids, and policy are accurate
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
    '''