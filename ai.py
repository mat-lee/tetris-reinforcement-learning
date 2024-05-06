from const import *
from game import Game

import json
import numpy as np
import os
import pandas as pd
import treelib

import keras
import tensorflow as tf

# For naming data and models
CURRENT_VERSION = 1.1

# Where data and models are saved
directory_path = '/Users/matthewlee/Documents/Code/Tetris Game'

# Areas of optimization:
# - Finding piece locations (piece locations held and not held are related)
# - Optimizing the search tree algorithm
# - Optimizing piece coordinates
# - Redundant piece rotations (Pick random one)

# AI todo:
# - Better network (e.g. L2, CNNs. more weights)
# - Temperature
# - Encoding garbage into the neural network
# - Encoding some sense of turn based into the network

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
        self.N = 1
        self.W = 0
        self.Q = 0
        self.P = 0

def MCTS(game, network):
    # Class for picking a move for the AI to make 
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

        # Go down the tree using Q+U until you get to a leaf node
        while not node.is_leaf():
            child_ids = node.successors(tree.identifier)
            max_child_score = -1
            max_child_id = None
            sum_n = 0
            for child_id in child_ids:
                sum_n += tree.get_node(child_id).data.N
            for child_id in child_ids:
                child_data = tree.get_node(child_id).data
                child_score = child_data.Q + child_data.P*sum_n/(1+child_data.N)
                if child_score >= max_child_score:
                    max_child_score = child_score
                    max_child_id = child_id
            
            node = tree.get_node(max_child_id)
            node_state = node.data

            DEPTH += 1
            if DEPTH > MAX_DEPTH:
                MAX_DEPTH = DEPTH

        # Place pieces if not the root node
        if not node.is_root():
            prior_node = tree.get_node(node.predecessor(tree.identifier))
            
            game_copy = prior_node.data.game.copy()
            node_state.game = game_copy
            node_state.game.make_move(node_state.move, add_bag=False, add_history=False)

        # Don't update policy, move_list, or generate new nodes if the node is done        
        if node_state.game.is_terminal == False:
            value, policy = evaluate(node_state.game, network)

            if node_state.game.no_move == False:
                move_matrix = get_move_matrix(node_state.game.players[node_state.game.turn])
                move_list = get_move_list(move_matrix, policy)

                # Generate new leaves
                for policy, move in move_list:
                    new_state = NodeState(game=None, move=move)
                    new_state.P = policy

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

        # Go back up the tree and updates nodes
        while not node.is_root():
            node_state = node.data
            node_state.N += 1
            node_state.W += value
            node_state.Q = node_state.W / node_state.N

            upwards_id = node.predecessor(tree.identifier)
            node = tree.get_node(upwards_id)

    print(MAX_DEPTH)

    # Choose a move
    root = tree.get_node("root")
    root_children_id = root.successors(tree.identifier)
    max_n = 0
    max_id = None

    for root_child_id in root_children_id:
        root_child = tree.get_node(root_child_id)
        root_child_n = root_child.data.N
        if root_child_n > max_n:
            max_n = root_child_n
            max_id = root_child.identifier

    move = tree.get_node(max_id).data.move

    return move, tree

def get_move_matrix(player): # e^-3
    # Using a list of all possible locations the piece can reach
    # Using a queue to determine which boards to check

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
            next_location_queue = []

            # Start the piece at the highest point it can be placed
            highest_row = get_highest_row(sim_player.board.grid)
            starting_row = max(highest_row - len(piece.matrix), ROWS - SPAWN_ROW)
            piece.y_0 = starting_row

            x = piece.x_0
            y = piece.y_0
            o = piece.rotation

            piece.update_rotation()

            next_location_queue.append((x, y, o))

            # Search through the queue
            while len(next_location_queue) > 0:
                piece.x_0, piece.y_0, piece.rotation = next_location_queue[0]

                piece.update_rotation()

                possible_piece_locations[piece.y_0][piece.x_0 + buffer][piece.rotation] = 1

                for move in [[1, 0], [-1, 0], [0, 1]]:
                    x = piece.x_0 + move[0]
                    y = piece.y_0 + move[1]
                    o = piece.rotation

                    if sim_player.can_move(x_offset=move[0], y_offset=move[1]): # Check this first to avoid index errors
                        if (possible_piece_locations[y][x + buffer][o] == 0 
                            and (x, y, o) not in next_location_queue
                            and y >= 0): # Avoid negative indexing

                            next_location_queue.append((x, y, o))


                for i in range(1, 4):
                    sim_player.try_wallkick(i)

                    x = piece.x_0
                    y = piece.y_0
                    o = piece.rotation

                    if (possible_piece_locations[y][x + buffer][o] == 0
                        and (x, y, o) not in next_location_queue
                        and y >= 0): # Avoid negative indexing
                        next_location_queue.append((x, y, o))

                    piece.x_0, piece.y_0, piece.rotation = next_location_queue[0]

                    piece.update_rotation()

                next_location_queue.pop(0)

            # Remove entries that can move downwards
            for o in range(4): # Smallest number of operations
                sim_player.piece.rotation = o
                sim_player.piece.update_rotation()

                for x in range(COLS + buffer - 1):
                    sim_player.piece.x_0 = x - buffer

                    for y in reversed(range(ROWS - 1)):
                        if possible_piece_locations[y][x][o] == 1:
                            sim_player.piece.y_0 = y
                            if not sim_player.can_move(y_offset=1):
                                all_moves[h][y][x][o] = 1

    return all_moves

def get_move_list(move_matrix, policy_matrix):
    # Returns list of possible moves sorted by policy
    move_list = []

    mask = np.multiply(move_matrix, policy_matrix)
    for h in range(len(mask)):
        for row in range(len(mask[0])):
            for col in range(len(mask[0][0])):
                for o in range(len(mask[0][0][0])):
                    if mask[h][row][col][o] > 0:
                        move_list.append((mask[h][row][col][o],
                                          (h, col - 2, row, o) # Switch to x y z and remove buffer
                                          ))
    # Sort by policy
    # move_list = sorted(move_list, key=lambda tup: tup[0], reverse=True)
    return move_list


def search_statistics(tree):
    """Return a matrix of proportion of moves looked at.

    Equal to N / Total nodes looked at for each move,
    and will become the target for training policy.
    # Policy: Rows x Columns x Rotations x Hold"""
    # Policy: 25 x 11 x 4 x 2
    probability_matrix = np.zeros((2, ROWS, COLS + 1, 4))

    root = tree.get_node("root")
    root_children_id = root.successors(tree.identifier)
    total_n = 0

    for root_child_id in root_children_id:
        root_child = tree.get_node(root_child_id)
        root_child_n = root_child.data.N
        root_child_move = root_child.data.move
        hold, col, row, rotation = root_child_move
        probability_matrix[hold][row][col][rotation] = root_child_n

        total_n += root_child_n
    
    probability_matrix /= total_n
    return probability_matrix.tolist()

# Using deepcopy:                   100 iter in 36.911 s
# Using copy functions in classes:  100 iter in 1.658 s
# Many small changes:               100 iter in 1.233 s
# MCTS uses game instead of player: 100 iter in 1.577 s

##### Neural Network #####

# Model Architecture 1.0:
# Player orientation: Active player, then other player
# X: 
#   (20 x 10) x 2 (Bool: Rows x Columns) Both players
#   (7 x 7) x 2   (Active piece,  held piece, closest to furthest queue pieces)
#                 (Bool: 1 or 0 for each piece type) Both players
#                 (ZLOSIJT)
#   (1) x 2       (Int: B2b) Both players
#   (1) x 2       (Int: Combo) Both Players
#   (1)           (Bool: Turn, 0 for first, 1 for second) 
#   (1)           (Int: Total pieces placed)
# y:
#   Policy: (2 x 25 x 11 x 4) = 2200 (Hold x Rows x Columns x Rotations)
#   Value: (1)
#
# 451 Total input values

class DataManager():
    def __init__(self) -> None:
        self.features_list = {
            "shape": [(25, 10, 1), (25, 10, 1), (7, 7), (7, 7), (1,), (1,), (1,), (1,), (1,), (1,)]
        }
    
    def create_input_layers(self):
        # Use a CNN for grids
        inputs = []
        flattened_inputs = []
        for i, shape in enumerate(self.features_list["shape"]):
            inputs.append(keras.Input(shape=shape))
            if shape == (25, 10, 1):
                x = keras.layers.Conv2D(256, (3, 3), padding="same")(inputs[i])
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Activation('relu')(x)
                flattened_inputs.append(keras.layers.Flatten()(x))
            else:
                flattened_inputs.append(keras.layers.Flatten()(inputs[i]))
        
        return inputs, flattened_inputs

    def ResidualLayer(self):
        def inside(x):
            x = keras.layers.Dense(256)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            return x
        return inside
    
def create_network(manager):
    # For now, jumble everything together
    inputs, flattened_inputs = manager.create_input_layers()

    x = keras.layers.Concatenate(axis=-1)(flattened_inputs)
    for i in range(2):
        x = manager.ResidualLayer()(x)

    value_output = keras.layers.Dense(1, activation='tanh')(x)
    policy_output = keras.layers.Dense(POLICY_SIZE, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])
    model.compile(optimizer='adam', loss=["mean_squared_error", "binary_crossentropy"])

    model.summary()

    #model.fit(x=grids, y=[values, policies])

    model.save(f"networks/{CURRENT_VERSION}.0.keras")

def train_network(model, data):
    features = list(map(list, zip(*data)))

    # Make features into np arrays
    for i in range(len(features)):
        features[i] = np.array(features[i])

    # Last two columns are value and policy
    policies = features.pop()
    values = features.pop()

    # Reshape policies
    policies = np.array(policies).reshape((-1, POLICY_SIZE))

    model.fit(x=features, y=[values, policies])

def evaluate(game, network):
    data = game_to_X(game)
    X = []
    for feature in data:
        X.append(np.expand_dims(np.array(feature), axis=0))

    values, policies = network.predict(X, verbose=0)
    policies = np.array(policies)
    policies = policies.reshape((2, ROWS, COLS+1, 4))
    return values, policies.tolist()

##### Simulation #####

# Having two AI's play against each other to generate a self-play dataset.
# At each move, save X and y for NN

def simplify_grid(grid):
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != 0:
                grid[row][col] = 1
    return grid

def game_to_X(game):
    def get_grids(game):
        grids = [[], []]

        for i, player in enumerate(game.players):
            grids[i] = [x[:] for x in player.board.grid] # Copy
            simplify_grid(grids[i])
        
        return grids
     
    def get_pieces(game):
        minos = "ZLOSIJT"
        piece_matrix = np.zeros((2, 7, 7))
        for i, player in enumerate(game.players):
            if player.piece != None: # Active piece: 0
                piece_matrix[i][0][minos.index(player.piece.type)] = 1
            if player.held_piece != None: # Held piece: 1
                piece_matrix[i][1][minos.index(player.held_piece)] = 1
            # Limit previews
            for j in range(min(len(player.queue.pieces), 5)): # Queue pieces: 2-6
                piece_matrix[i][j + 2][minos.index(player.queue.pieces[j])] = 1

        return piece_matrix
    
    def get_b2b(game):
        b2b = [player.stats.b2b for player in game.players]
        return b2b
    
    def get_combo(game):
        combo = [player.stats.combo for player in game.players]
        return combo

    grids = get_grids(game)
    pieces = get_pieces(game)
    b2b = get_b2b(game)
    combo = get_combo(game)
    color = game.players[game.turn].color
    pieces_placed = game.players[game.turn].stats.pieces

    return grids[0], grids[1], pieces[0], pieces[1], b2b[0], b2b[1], combo[0], combo[1], color, pieces_placed

def play_game(network, NUMBER, show_game=False):
    # AI plays one game against itself
    # Returns: grids, pieces, first move, pieces placed, value, policy
    if show_game == True:
        screen = pygame.display.set_mode( (WIDTH, HEIGHT))
        pygame.display.set_caption(f'Training game {NUMBER}')

    game = Game()
    game.setup()

    # Initialize data storage
    # Each index is a player
    game_data = [[], []]

    while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
        move, tree = MCTS(game, network)
        probability_matrix = search_statistics(tree)

        move_data = [*game_to_X(game)]
        # Convert to regular lists
        for i in range(len(move_data)):
            if isinstance(move_data[i], np.ndarray):
                move_data[i] = move_data[i].tolist()
        
        move_data.append(probability_matrix)
        game_data[game.turn].append(move_data)

        game.make_move(move)

        if show_game == True:
            game.show(screen)
            pygame.display.update()
    
    # After game ends update value
    winner = game.winner
    for i in range(len(game_data)): # 
        if winner == -1:
            value = 0
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

def make_training_set(network, num_games):
    series_data = []
    for i in range(num_games):
        data = play_game(network, i, show_game=True)
        series_data.extend(data)

    json_data = json.dumps(series_data)

    # Increment set counter
    next_set = get_highest_number('data') + 1

    with open(f"data/{CURRENT_VERSION}.{next_set}.txt", 'w') as out_file:
        out_file.write(json_data)

def training_loop(manager, network):
    for i in range(TRAINING_LOOPS):
        make_training_set(network, TRAINING_GAMES)
        print("Finished set")

        data = []

        # Load data from the past n games
        # Find highest set number
        max_set = get_highest_number('data')
        
        # Get n games
        for filename in os.listdir('data'):
            if filename[:3] == str(CURRENT_VERSION):
                number = int(filename[4:-4])
                if number > max_set - 10:
                    # Load data
                    set = json.load(open(f"data/{filename}", 'r'))
                    data.extend(set)

        train_network(network, data)
    print("Finished loop")

def battle_networks(NN_1, NN_2, show_game=False):
    wins = np.zeros((2))
    for i in range(BATTLE_GAMES):
        if show_game == True:
            screen = pygame.display.set_mode( (WIDTH, HEIGHT))
            pygame.display.set_caption('Battle game')

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

    wins /= wins.sum()
    return wins[0], wins[1]

def self_play_loop(network, manager=DataManager()):
    old_network = network
    iter = 0
    while True:
        iter += 1
        new_network = keras.models.clone_model(old_network)
        
        training_loop(manager, new_network)

        new_wins, old_wins = battle_networks(new_network, old_network, show_game=True)
        print("Finished battle")
        if new_wins >= 0.55:
            old_network = new_network

            next_ver = get_highest_number('networks') + 1

            old_network.save(f"networks/{CURRENT_VERSION}.{next_ver}.keras")
        else:
            break

def load_best_network():
    max_ver = get_highest_number('networks')

    path = f'networks/{CURRENT_VERSION}.{max_ver}.keras'

    return tf.keras.models.load_model(path)

def get_highest_number(folder):
    max = 0
    for filename in os.listdir(folder):
        if filename[:3] == str(CURRENT_VERSION):
            filename = filename[4:]
            number = int(filename.split('.', 1)[0])
            if number > max:
                max = number
    return max