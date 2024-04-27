from const import *
from game import Game
from player import Player

import json
import numpy as np
import pickle
import random
import treelib

import keras
import tensorflow as tf

# Areas of optimization:
# - Finding piece locations (piece locations held and not held are related)
# - Optimizing the search tree algorithm
# - Optimizing piece coordinates

class NodeState():
    """Node class for storing the game in the tree.
    
    Similar to the game class, but is more specialized for AI functions and has different variables"""

    def __init__(self, game=None, move=None) -> None:
        self.game = game
        self.move = move

        # Search tree variables
        # Would be stored in each edge, but this is essentially the same thing
        self.N = 1
        self.W = 0
        self.Q = 0
        self.P = 0

        self.value = None
        self.policy = None
        self.move_list = None

def MCTS(game, network):
    # Class for picking a move for the AI to make 
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
    while iter < MAX_ITER:
        iter += 1

        # Begin at the root node
        node = tree.get_node("root")
        node_state = node.data

        DEPTH = 0

        # Go down the tree using Q+U until you get to a leaf node
        while not node.is_leaf():
            child_ids = node.successors(tree.identifier)
            max_child_score = 0
            max_child_id = None
            sum_n = 0
            for child_id in child_ids:
                sum_n += tree.get_node(child_id).data.N
            for child_id in child_ids:
                child_data = tree.get_node(child_id).data
                child_score = child_data.P*sum_n/(1+child_data.N)
                if child_score > max_child_score:
                    max_child_score = child_score
                    max_child_id = child_id
            
            node = tree.get_node(max_child_id)
            node_state = node.data

            DEPTH += 1
            if DEPTH > MAX_DEPTH:
                MAX_DEPTH = DEPTH
                    
        # Don't update policy, move_list, or generate new nodes if the node is done
        if node_state.game.is_terminal == False:

            value, policy = evaluate(node_state.game, network)
            move_matrix = get_move_matrix(node_state.game.players[node_state.game.turn])
            move_list = get_move_list(move_matrix, policy)

            # Place pieces and generate new leaves
            for policy, move in move_list:
                game_copy = node_state.game.copy()
                new_state = NodeState(game=game_copy, move=move)

                new_state.game.make_move(move, add_bag=False, add_history=False)
                new_state.P = policy

                tree.create_node(data=new_state, parent=node.identifier)

        # Go back up the tree and updates nodes
        while not node.is_root():
            upwards_id = node.predecessor(tree.identifier)
            node = tree.get_node(upwards_id)

            node_state = node.data
            node_state.N += 1
            node_state.W += value
            node_state.Q = node_state.W / node_state.N

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
    for h in range(2):
        # Load the state
        sim_player = player.copy()
        if h == 1:
            sim_player.hold_piece()

        piece = sim_player.piece

        # No piece can be placed in the bottom row; ROWS - 1
        # possible_piece_locations = [[[False for o in range (4)] for x in range(COLS + buffer - 1)] for y in range(ROWS - 1)]
        possible_piece_locations = np.zeros((ROWS, width, 4))
        next_location_queue = []
        locations = []

        # Start the piece at the highest point it can be placed
        highest_row = get_highest_row(sim_player.board.grid)
        starting_row = max(highest_row - len(piece.matrix), 0)
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
    move_list = sorted(move_list, key=lambda tup: tup[0], reverse=True)
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

##### Simulation #####

# Having two AI's play against each other.
# At each move, save X and y for NN
# X: 
#   (20 x 10 x 2) (Rows x Columns x Boards)
# y:
#   Policy: (2 x 25 x 11 x 4) = 2200 (Hold x Rows x Columns x Rotations)
#   Value: (1)

def simplify_grid(grid):
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != 0:
                grid[row][col] = 1
    return grid

def play_game():
    # Player data: (Boards, (Value, Policy))
    game = Game()
    game.setup()
    player_data = [[], []]

    while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
        move, tree = MCTS(game)
        # Boards from the perspective of the active player
        # Also make it 1s and 0s
        grids = [[], []]
        for i, player in enumerate(game.players):
            grids[i] = [x[:] for x in player.board.grid]
            simplify_grid(grids[i])

        if game.turn == 1:
            grids = grids[::-1]
        probability_matrix = search_statistics(tree)

        game.make_move(move)
        player_data[game.turn].append((grids, probability_matrix))
    
    # Check if someone won
    # 0: Draw, 1: Player 1, 2: Player 2
    winner = game.winner
    for i in range(len(player_data)):
        if winner == 0:
            player_data[i] = [(grids, (0, policy)) for grids, policy in player_data[i]]
        elif winner == i:
            player_data[i] = [(grids, (1, policy)) for grids, policy in player_data[i]]
        else:
            player_data[i] = [(grids, (-1, policy)) for grids, policy in player_data[i]]

    # Reformat data
    data = player_data[0]
    data.extend(player_data[1])

    json_data = json.dumps(data)
    with open("data/out.txt", 'w') as out_file:
        out_file.write(json_data)

def features_targets(data):
    boards = []
    values = []
    policies = []
    for i in range(len(data)):
        boards.append(data[i][0])
        values.append(data[i][1][0])
        policies.append(data[i][1][1])
    
    return boards, (values, policies)

def create_network(data):
    grids, (values, policies) = features_targets(data)
    grids = np.array(grids)
    values = np.array(values)
    policies = np.array(policies)
    # Reshape policies
    policies = policies.reshape((-1, POLICY_SIZE))

    inputs = keras.Input(shape=(2, 25, 10))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(16, activation="relu")(x)
    value_output = keras.layers.Dense(1)(x)
    policy_output = keras.layers.Dense(POLICY_SIZE, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])
    model.compile(optimizer='adam', loss=["mean_squared_error", "binary_crossentropy"])

    model.summary()

    model.fit(x=grids, y=[values, policies])

    model.save("networks/model1.keras")

def evaluate(game, network):
    grids = [[], []]
    for i, player in enumerate(game.players):
        grids[i] = [x[:] for x in player.board.grid]
        simplify_grid(grids[i])

    if game.turn == 1:
        grids = grids[::-1]
    
    X = np.array(grids)
    X = np.expand_dims(X, axis=0)

    values, policies = network.predict(X, verbose=0)
    policies = np.array(policies)
    policies = policies.reshape((2, ROWS, COLS+1, 4))
    return values, policies.tolist()

def load_best_network():
    return tf.keras.models.load_model('networks/model1.keras')

if __name__ == "__main__":
    data = json.load(open("data/1.1G.NH.txt", 'r'))
    create_network(data)

    # value_model = LinearRegression()
    # policy_model = LinearRegression()
    # value_model.fit(boards, value)
    # policy_model.fit(boards, policy)

    # with open('networks/value_model.pkl','wb') as f:
    #     pickle.dump(value_model,f)
    # with open('networks/policy_model.pkl','wb') as f:
    #     pickle.dump(value_model,f)

# I can't figure out how to install tensorflow onto my Macbook

# inputs = keras.input(shape=(20, 10, 2))
# x = keras.Dense(16, activation="relu")(inputs)
# value_output = keras.Dense(1)(x)
# policy_output = keras.Dense((25, 11, 4, 2))(x)

# model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])
# model.compile(optimizer='adam', loss=["mean_squared_error", "binary_crossentropy"])

# model.fit()