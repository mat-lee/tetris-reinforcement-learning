from const import *
from game import Game

import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import treelib

import keras
import tensorflow as tf
from tensorflow.python.ops import math_ops
from sklearn.model_selection import train_test_split

import cProfile
import pstats

# For naming data and models
CURRENT_VERSION = 2.0

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
    for _ in range(MAX_ITER):
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
            sum_visits = 0
            for child_id in child_ids:
                sum_visits += tree.get_node(child_id).data.visit_count
            for child_id in child_ids:
                # For each child calculate a score
                # Polynomial upper confidence trees (PUCT)
                child_data = tree.get_node(child_id).data
                child_score = child_data.value_avg + child_data.policy*math.sqrt(sum_visits)/(1+child_data.visit_count)
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
            # value, policy = evaluate(node_state.game, network)
            value, policy = random_evaluate()

            if node_state.game.no_move == False:
                move_matrix = get_move_matrix(node_state.game.players[node_state.game.turn])
                move_list = get_move_list(move_matrix, policy)

                # Generate new leaves
                # Normalize policy values
                policy_sum = sum(list(zip(*move_list))[0])
                # sum(policy for policy, move in movelist)

                for policy, move in move_list:
                    new_state = NodeState(game=None, move=move)
                    new_state.policy = policy / policy_sum

                    tree.create_node(data=new_state, parent=node.identifier)
        
        # Node is terminal
        # Update weights based on winner
        else: 
            winner = node_state.game.winner
            if winner == 0: # AI Starts
                value = 1
            elif winner == 1: # Opponent
                value = -1
            else:
                value = 0

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

        # Go back up the tree and updates nodes
        # Propogate positive values for the player made the move, and negative for the other player
        final_node_turn = node_state.game.turn

        while not node.is_root():
            node_state = node.data
            node_state.visit_count += 1
            # Revert value if the other player just went
            node_state.value_sum += (value if node_state.game.turn == final_node_turn else -value)
            node_state.value_avg = node_state.value_sum / node_state.visit_count

            upwards_id = node.predecessor(tree.identifier)
            node = tree.get_node(upwards_id)

    #print(MAX_DEPTH)

    # Choose a move based on the number of visits
    root = tree.get_node("root")
    root_children_id = root.successors(tree.identifier)
    max_n = 0
    max_id = None

    for root_child_id in root_children_id:
        root_child = tree.get_node(root_child_id)
        root_child_n = root_child.data.visit_count
        if root_child_n > max_n:
            max_n = root_child_n
            max_id = root_child.identifier

    move = tree.get_node(max_id).data.move

    return move, tree

def get_move_matrix(player): # e^-3
    # Returns a list of all possible moves that a player can make
    # Very Important: With my coordinate system, pieces can be placed with negative x values
    # To avoid negative indexing, the list of moves is shifted by 2
    # It encodes moves from -2 to 8 as indeces 0 to 10

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

    # Repeat for held pieces
    all_moves = np.zeros((2, ROWS, width, 4))

    # Load the state
    sim_player = player.copy()

    for h in range(2): # Look through current piece and held piece
        if h == 1:
            sim_player.hold_piece()

        piece = sim_player.piece
        if piece != None: # Skip if there's no piece
            # No piece can be placed under the grid >= ROWS - 1

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

                for x in range(COLS + buffer - 1):
                    sim_player.piece.x_0 = x - buffer

                    for y in reversed(range(ROWS - 1)):
                        if possible_piece_locations[y][x][o] == 1:
                            sim_player.piece.y_0 = y
                            piece.coordinates = piece.get_self_coords
                            if not sim_player.can_move(piece, y_offset=1):
                                all_moves[h][y][x][o] = 1

    return all_moves

def get_move_list(move_matrix, policy_matrix):
    # Returns list of possible moves with their policy
    # Removes buffer
    move_list = []

    mask = np.multiply(move_matrix, policy_matrix)
    # Ordered from smallest to largest
    for h in range(len(mask)):
        for o in range(len(mask[0][0][0])):
            for col in range(len(mask[0][0])):
                for row in range(len(mask[0])):
                    if mask[h][row][col][o] > 0:
                        move_list.append((mask[h][row][col][o],
                                          (h, col - 2, row, o) # Switch to x y z and remove buffer
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

##### Neural Network #####

# Model Architecture 1.2:
# Player orientation: Active player, other player
# X: 
#   (20 x 10) x 2 (Bool: Rows x Columns) Both players
#   (7 x 7) x 2   (Active piece,  held piece, closest to furthest queue pieces)
#                 (Bool: 1 or 0 for each piece type) Both players
#                 (ZLOSIJT)
#   (1) x 2       (Int: B2b) Both players
#   (1) x 2       (Int: Combo) Both Players
#   (1) x 2       (Int: Lines cleared) Both Players
#   (1) x 2       (Int: Lines sent) Both Players
#   (1)           (Bool: Turn, 0 for first, 1 for second) 
#   (1)           (Int: Total pieces placed)
# y:
#   Policy: (2 x 25 x 11 x 4) = 2200 (Hold x Rows x Columns x Rotations)
#   Value: (1)
#
# 508 Total input values
# 
# Model <1.7: In the trenches
# 
# Model 1.8: Added data augmentation and changed amount of training data and added dirichlet noise: 
# Performs semi rare line clears
# 
# Model 1.9: Fixed data augmentation pieces not being swapped correctly
# Back in the trenches
#
# Model 2.0: Normalized policy values

class DataManager():
    # Handles layer sizes
    def __init__(self) -> None:
        self.features_list = {
            "shape": [(25, 10, 1), (25, 10, 1), 
                      (7, 7), (7, 7), 
                      (1,), (1,), 
                      (1,), (1,), 
                      (1,), (1,), 
                      (1,), (1,), 
                      (1,),       
                      (1,)]       
        }
    
    def create_input_layers(self):
        # Use a convolutional layer
        inputs = []
        flattened_inputs = []
        for i, shape in enumerate(self.features_list["shape"]):
            inputs.append(keras.Input(shape=shape))
            if shape == self.features_list["shape"][0]:
                x = keras.layers.Conv2D(32, (3, 3), padding="valid")(inputs[i])
                # x = keras.layers.MaxPooling2D( 
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Activation('relu')(x)
                flattened_inputs.append(keras.layers.Flatten()(x))
            else:
                flattened_inputs.append(keras.layers.Flatten()(inputs[i]))
        
        return inputs, flattened_inputs

    def ResidualLayer(self):
        # Uses skip connections
        def inside(x):
            y = keras.layers.Dense(32)(x)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Activation('relu')(y)
            y = keras.layers.Dense(32)(y)
            y = keras.layers.BatchNormalization()(y)
            z = keras.layers.Add()([x, y]) # Skip connection
            z = keras.layers.Activation('relu')(z)

            return z
        return inside

    def ValueHead(self):
        def inside(x):
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.Dense(32)(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.Dense(1, activation='tanh')(x)

            return x
        return inside

    def PolicyHead(self):
        def inside(x):
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            # Generate probability distribution
            x = keras.layers.Dense(POLICY_SIZE, activation="softmax")(x)

            return x
        return inside

def create_network(manager):
    # Creates a network with random weight
    # Input layer: Applies a convolution to the grids and flattens all inputs
    # -> Fully connected layer to adjust data size
    # -> Three residual layers
    # -> Value head and policy head

    # , kernel_regularizer=keras.regularizers.L2(1e-4)

    inputs, flattened_inputs = manager.create_input_layers()

    x = keras.layers.Concatenate(axis=-1)(flattened_inputs)

    # Fully connected layer
    x = keras.layers.Dense(32)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    for _ in range(3):
        x = manager.ResidualLayer()(x)
        #x = keras.layers.Dense(32, activation='relu')(x)

    value_output = manager.ValueHead()(x)
    policy_output = manager.PolicyHead()(x)

    model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])
    # Loss is the sum of MSE of values and Cross entropy of policies
    model.compile(optimizer='adam', loss=["mean_squared_error", "categorical_crossentropy"], loss_weights=[1, 1])

    model.summary()

    #model.fit(x=grids, y=[values, policies])

    model.save(f"{directory_path}/{CURRENT_VERSION}.0.keras")

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
        X.append(np.expand_dims(np.array(feature), axis=0))

    values, policies = network.predict(X, verbose=0)
    policies = np.array(policies)
    policies = policies.reshape((2, ROWS, COLS+1, 4))
    return values, policies.tolist()

def random_evaluate():
    # For testing how fast the MCTS is
    return random.random() * 2 -1, np.ones((2, ROWS, COLS+1, 4)).tolist()

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
    probability_matrix = np.zeros((2, ROWS, COLS + 1, 4))

    root = tree.get_node("root")
    root_children_id = root.successors(tree.identifier)
    total_n = 0

    for root_child_id in root_children_id:
        root_child = tree.get_node(root_child_id)
        root_child_n = root_child.data.visit_count
        root_child_move = root_child.data.move
        hold, col, row, rotation = root_child_move
        probability_matrix[hold][row][col + 2][rotation] = root_child_n

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

def game_to_X(game):
    # Returns game information for the network.
    def get_grids(game):
        grids = [[], []]

        for i, player in enumerate(game.players):
            grids[i] = [x[:] for x in player.board.grid] # Copy
            simplify_grid(grids[i])
        
        if game.turn == 1: grids = grids[::-1]
        
        return grids
     
    def get_pieces(game):
        minos = "ZLOSIJT"
        piece_table = np.zeros((2, 7, 7), dtype=int)
        for i, player in enumerate(game.players):
            if player.piece != None: # Active piece: 0
                piece_table[i][0][minos.index(player.piece.type)] = 1
            if player.held_piece != None: # Held piece: 1
                piece_table[i][1][minos.index(player.held_piece)] = 1
            # Limit previews
            for j in range(min(len(player.queue.pieces), 5)): # Queue pieces: 2-6
                piece_table[i][j + 2][minos.index(player.queue.pieces[j])] = 1

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

    # Orient all info in perspective to the current player
    grids = get_grids(game)
    pieces = get_pieces(game)
    b2b = get_b2b(game)
    combo = get_combo(game)
    lines_cleared = get_lines_cleared(game)
    lines_sent = get_lines_sent(game)
    color = game.players[game.turn].color
    pieces_placed = game.players[game.turn].stats.pieces

    return grids[0], grids[1], pieces[0], pieces[1], b2b[0], b2b[1], combo[0], combo[1], lines_cleared[0], lines_cleared[1], lines_sent[0], lines_sent[1], color, pieces_placed

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
    swap_dict = {
        0: 3,
        3: 0,
        1: 5,
        5: 1
    }

    reflected_piece_table = np.zeros((7, 7), dtype=int)

    for i, piece_row in enumerate(piece_table):
        if 1 in piece_row:
            piece_index = piece_row.tolist().index(1) # Index of the piece in ZLOSIJT
            if piece_index in swap_dict:
                # Swap piece if swappable
                piece_index = swap_dict[piece_index]
            reflected_piece_table[i][piece_index] = 1
    
    return reflected_piece_table

def reflect_policy(policy_matrix, active_piece_size, hold_piece_size):
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
    # AI plays one game against itself and generates data.
    # Returns: grids, pieces, first move, pieces placed, value, policy
    if show_game == True:
        screen = pygame.display.set_mode( (WIDTH, HEIGHT))
        pygame.display.set_caption(f'Training game {NUMBER}')

    game = Game()
    game.setup()

    # Initialize data storage
    # Each index is a player
    game_data = [[], []]

    # Model 2.0 with 20 iter takes 2.579 seconds in debug mode
    while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
        # with cProfile.Profile() as pr:
        move, tree = MCTS(game, network, add_noise=True)
        search_matrix = search_statistics(tree)

        # Piece sizes of players are needed to find where a reflected piece ends up
        active_piece_size, hold_piece_size = get_piece_sizes(game.players[game.turn])
        reflected_search_matrix = reflect_policy(search_matrix, active_piece_size, hold_piece_size)

        # Get data
        move_data = [*game_to_X(game)] 

        # Reflect each player for a total of 2 * 2 = 4 times more data
        # Reflect grid, pieces, and policy
        for active_player_idx in range(2): # Reflect or not active player
            for other_player_idx in range(2):# Reflect or not other player
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
                    copied_data[2] = reflect_pieces(copied_data[2])
                
                if other_player_idx == 1:
                    copied_data[1] = reflect_grid(copied_data[1])
                    copied_data[3] = reflect_pieces(copied_data[3])

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
            value = 0
        # Set values to 1 and -1
        elif winner == i:
            value = 1
        else:
            value = -1
        # Insert value before policy
        for j in range(len(game_data[i])):
            game_data[i][j].insert(-1, value)

    # Reformat data
    data = game_data[0]
    data.extend(game_data[1])

    return data

def make_training_set(network, num_games, show_game=False):
    # Creates a dataset of several AI games.
    series_data = []
    for i in range(num_games):
        data = play_game(network, i, show_game=show_game)
        series_data.extend(data)

    json_data = json.dumps(series_data)

    # Increment set counter
    next_set = get_highest_number('.txt') + 1

    with open(f"{directory_path}/{CURRENT_VERSION}.{next_set}.txt", 'w') as out_file:
        out_file.write(json_data)

def training_loop(network, show_games=False):
    # Play a training set and train the network on past sets.
    for i in range(TRAINING_LOOPS):
        make_training_set(network, TRAINING_GAMES, show_game=show_games)
        print("Finished set")

        data = []

        # Load data from the past n games
        # Find highest set number
        max_set = get_highest_number('.txt')
        
        # Get n games
        for filename in get_filenames('.txt'):
            number = int(filename[4:-4])
            if number > max_set - SETS_TO_TRAIN_WITH:
                # Load data
                set = json.load(open(f"{directory_path}/{filename}", 'r'))
                data.extend(set)
    
        # Train network on a fraction of the available data
        data_batch, _ = train_test_split(data, train_size=0.1)
        print(len(data), len(data_batch))
        train_network(network, data_batch)
    print("Finished loop")

def battle_networks(NN_1, NN_2, show_game=False):
    # Battle two AI's with different networks.
    wins = np.zeros((2), dtype=int)
    for i in range(BATTLE_GAMES):
        if show_game == True:
            screen = pygame.display.set_mode( (WIDTH, HEIGHT))
            pygame.display.set_caption(f'Battle game: {wins[0]} to {wins[1]}')

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

    print(*wins)

    wins = [win/sum(wins) for win in wins]
    return wins[0], wins[1]

def self_play_loop(network, show_games=False):
    if show_games == True:
        pygame.init()
    # Given a network, generates training data, trains it, and checks if it improved.
    old_network = network
    iter = 0
    while True:
        iter += 1
        new_network = keras.models.clone_model(old_network)
        
        training_loop(new_network, show_games=show_games)

        new_wins, old_wins = battle_networks(new_network, old_network, show_game=show_games)
        print("Finished battle")
        if new_wins >= 0.55:
            next_ver = get_highest_number('.keras') + 1
            new_network.save(f"{directory_path}/{CURRENT_VERSION}.{next_ver}.keras")
            # The new network becomes the network to beat
            old_network = new_network
        else:
            break

def load_best_network():
    max_ver = get_highest_number('.keras')

    path = f"{directory_path}/{CURRENT_VERSION}.{max_ver}.keras"

    return tf.keras.models.load_model(path)

def get_highest_number(extension):
    max = 0
    for filename in get_filenames(extension):
        filename = filename[4:]
        number = int(filename.split('.', 1)[0])
        if number > max:
            max = number
    return max

def get_filenames(extension):
    filenames = []

    for filename in os.listdir(f'{directory_path}'):
        if filename.startswith(str(CURRENT_VERSION)) and filename.endswith(extension):
            filenames.append(filename)
    
    return filenames