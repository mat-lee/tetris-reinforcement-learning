from architectures import *
from const import *
from game import Game

import copy
from collections import deque
import gc
import itertools
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
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from tensorflow import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import math_ops
from sklearn.model_selection import train_test_split

# Reduce tensorflow text
# tf.get_logger().setLevel('ERROR')
# tf.autograph.set_verbosity(0)

from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Not sure where I'm importing pygame from but sure
pygame.init()

import cProfile
import pstats

# For naming data and models
MODEL_VERSION = 5.0
DATA_VERSION = 2.0

# Where data and models are saved
directory_path = '/Users/matthewlee/Documents/Code/Tetris Game/Storage'

# Areas of optimization:
# - Generating move matrix
# - Switching to pytorch (?)
# - Load data faster

# AI todo:
# - Test parameters
# - Finish pytorch merger, clean up merge code, test
# - Encoding garbage into the neural network
# - Investigate hold issue (maybe search depth?)
# - Add changing learning rate
# - Keep implementing katago strategies (read appendix)
#     - Randomize first r moves
#     - Changing board sizes

'''
# Testing results

l1_neurons
l2_neurons

# Alphalike model
blocks
pooling_blocks
filters
cpool
dropout
    - 0 < 0.25 > 0.4
kernels=1,
o_side_neurons=16,
value_head_neurons=16,

augment_data
    - True > False
learning_rate
    - When training on a single set ??? (Don't know how to change in keras after initialization)
loss_weights
epochs
batch_size

save_all
    - False > True

use_playout_cap_randomization=
playout_cap_chance=
playout_cap_mult=

use_dirichlet_noise
DIRICHLET_ALPHA
DIRICHLET_S
    - 'DIRICHLET_ALPHA' == 0.01: 10 > 50
DIRICHLET_EXPLORATION
use_dirichlet_s

FpuStrategy
    'reduction' > absolute
FpuValue
    'reduction': 0.4 ~= 0.2 > 0.0

use_forced_playouts_and_policy_target_pruning
CForcedPlayout

use_root_softmax
RootSoftmaxTemp

CPUCT

Miscellaneous
 - With 10 layers, 16 filters is max number of filters before inference time scales
 - Bottleneck residuals were worse, with or without dropout than normal residuals (test_4 and test_5)
 - test_data_paramemeters does work

'''

total_branch = 0
number_branch = 0

class Config():
    def __init__(
        self, 

        model='keras',
        default_model=None,

        # Architecture Parameters
        # Fishlike model
        l1_neurons=256, 
        l2_neurons=32,

        # Alphalike model
        blocks=10,
        pooling_blocks=2,
        filters=32, 
        cpool=8,
        head_filters=8,

        # Only use one of dropout or l2_reg
        dropout=0.25,
        l2_reg=3e-5,

        kernels=1,
        o_side_neurons=16,
        value_head_neurons=16,

        use_tanh=False, # Affects data saving and model activation

        # Training Parameters
        augment_data=True,
        learning_rate=0.001, 
        loss_weights=[1, 1], 
        epochs=1, 
        batch_size=64,
        shuffle=True,

        # MCTS Parameters
        training=False, # Set to true to use a variety of features
        use_experimental_features=False, # Before setting to true, check if it's in use
        save_all=False,

        MAX_ITER=160, # 1600 ##########
        use_playout_cap_randomization=True,
        playout_cap_chance=0.25,
        playout_cap_mult=5,

        use_dirichlet_noise=True,
        DIRICHLET_ALPHA=0.01,
        DIRICHLET_S=25,
        DIRICHLET_EXPLORATION=0.25, 
        use_dirichlet_s=True,

        FpuStrategy='reduction', # 'reduction' subtracts FpuValue from parent eval, 'absolute' uses FpuValue
        FpuValue=0.4,

        use_forced_playouts_and_policy_target_pruning=True,
        CForcedPlayout=2,

        use_root_softmax=True,
        RootSoftmaxTemp=1.1,
        
        CPUCT=0.75
    ):
        self.model = model
        self.default_model = default_model
        self.l1_neurons = l1_neurons
        self.l2_neurons = l2_neurons
        self.blocks = blocks
        self.pooling_blocks = pooling_blocks
        self.filters = filters
        self.cpool = cpool
        self.head_filters = head_filters
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.kernels = kernels
        self.o_side_neurons = o_side_neurons
        self.value_head_neurons = value_head_neurons
        self.use_tanh = use_tanh
        self.augment_data = augment_data
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training = training
        self.use_experimental_features = use_experimental_features
        self.save_all = save_all
        self.MAX_ITER = MAX_ITER
        self.use_playout_cap_randomization = use_playout_cap_randomization
        self.playout_cap_chance = playout_cap_chance
        self.playout_cap_mult = playout_cap_mult
        self.use_dirichlet_noise = use_dirichlet_noise
        self.DIRICHLET_ALPHA = DIRICHLET_ALPHA
        self.DIRICHLET_S = DIRICHLET_S
        self.DIRICHLET_EXPLORATION = DIRICHLET_EXPLORATION
        self.use_dirichlet_s = use_dirichlet_s
        self.FpuStrategy = FpuStrategy
        self.FpuValue = FpuValue
        self.use_forced_playouts_and_policy_target_pruning = use_forced_playouts_and_policy_target_pruning
        self.CForcedPlayout = CForcedPlayout
        self.use_root_softmax = use_root_softmax
        self.RootSoftmaxTemp = RootSoftmaxTemp
        self.CPUCT = CPUCT

class NodeState():
    """Node class for storing the game in the tree.
    
    Similar to the game class, but is more specialized for AI functions and has different variables"""

    def __init__(self, game=None, move=None) -> None:
        self.game = game
        self.move = move

        # Search tree variables
        # Would be stored in each edge, but this is essentially the same thing

        self.visit_count = 0
        self.value_sum = 0
        self.value_avg = 0
        self.policy = 0

def MCTS(config, game, network, move_algorithm='faster-but-loss') -> tuple[tuple, treelib.Tree, bool]:
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

    max_iterations = None
    fast_iter = False # Used to disable dirichlet noise for fast iterations

    # Randomly assigns moves to higher and lower playouts
    if config.training and config.use_playout_cap_randomization:
        if random.random() < config.playout_cap_chance:
            max_iterations = math.ceil(config.playout_cap_mult * (config.MAX_ITER / (config.playout_cap_chance * (config.playout_cap_mult - 1) + 1)))
        else:
            max_iterations = math.floor(config.MAX_ITER / (config.playout_cap_chance * (config.playout_cap_mult - 1) + 1))
            fast_iter = True
    else:
        max_iterations = config.MAX_ITER

    if config.FpuStrategy == 'reduction':
        total_expanded_policy = 0

    while iter < max_iterations:
        iter += 1

        # Begin at the root node
        node = tree.get_node("root")
        node_state = node.data

        DEPTH = 0

        # Go down the tree using formula Q+U until you get to a leaf node
        # However, if using forced playouts, select a node if it has fewer than the forced playouts amount
        while not node.is_leaf():
            child_ids = node.successors(tree.identifier)
            max_child_score = -1
            max_child_id = None
            parent_visits = node.data.visit_count

            number_branch += 1
            
            # Debuging
            Qs = []
            Us = []

            # Look through each child
            for child_id in child_ids:
                
                total_branch += 1

                # For each child calculate a score
                # Polynomial upper confidence trees (PUCT)
                child_data = tree.get_node(child_id).data

                child_score = child_data.value_avg + config.CPUCT * child_data.policy*math.sqrt(parent_visits)/(1+child_data.visit_count)

                Qs.append(child_data.value_avg)
                Us.append(config.CPUCT* child_data.policy*math.sqrt(parent_visits)/(1+child_data.visit_count))

                # Check forced playouts
                if config.use_forced_playouts_and_policy_target_pruning and config.training: # Only use during training/when configured
                    if node.is_root(): # Only use for the root
                        if not (config.use_playout_cap_randomization == True and fast_iter == True): # Don't use during fast iterations
                            if child_data.visit_count >= 1:
                                n_forced = math.sqrt(config.CForcedPlayout * child_data.policy * parent_visits)

                                if child_data.visit_count < n_forced:
                                    child_score = float('inf')

                if child_score >= max_child_score:
                    max_child_score = child_score
                    max_child_id = child_id
            
            # Pick the node with the highest score
            node = tree.get_node(max_child_id)
            node_state = node.data

            DEPTH += 1
            if DEPTH > MAX_DEPTH:
                MAX_DEPTH = DEPTH
            
            if iter == config.MAX_ITER - 1:
                pass

        # If not the root node, place piece in node
        if not node.is_root():
            prior_node = tree.get_node(node.predecessor(tree.identifier))
            
            game_copy = prior_node.data.game.copy()
            node_state.game = game_copy
            node_state.game.make_move(node_state.move, add_bag=False, add_history=False)

            # Keep track of total expanded policy
            # Root node has no policy prior
            if config.FpuStrategy == 'reduction':
                total_expanded_policy += node_state.policy

        # Don't update policy, move_list, or generate new nodes if the game is over       
        if node_state.game.is_terminal == False:
            if config.model == 'keras':
                value, policy = evaluate_from_tflite(node_state.game, network)
            elif config.model == 'pytorch':
                value, policy = evaluate_pytorch(node_state.game, network)
            # value, policy = random_evaluate()

            if node_state.game.no_move == False:
                move_matrix = get_move_matrix(node_state.game.players[node_state.game.turn], algo=move_algorithm)
                move_list = get_move_list(move_matrix, policy)

                assert len(move_list) > 0 # There should always be a legal move, or the game would be over

                # Generate new leaves
                policies, moves = map(list, zip(*move_list))

                # Debugging
                # pn = []
                # ps = []
                # pn_s = sum(policies)

                # Apply softmax and temperature to policy of children of the root node
                if node.is_root() and config.use_root_softmax:
                    # Formula taken from katago
                    max_policy = max(policies)
                    for i in range(len(policies)):
                        # pn.append(policies[i])

                        policies[i] = math.exp((math.log(policies[i]) - math.log(max_policy)) * 1 / config.RootSoftmaxTemp) # ???
                        
                        # ps.append(policies[i])

                policy_sum = sum(policies)

                for policy, move in zip(policies, moves):
                    new_state = NodeState(game=None, move=move)

                    # New node policy
                    new_state.policy = policy / policy_sum # Normalize
                    assert new_state.policy > 0

                    # New node value
                    # Set First Play Urgency value
                    if config.FpuStrategy == 'absolute':
                        new_state.value_avg = max (0, config.FpuValue)
                    elif config.FpuStrategy == 'reduction':
                        # value is the parent node's value
                        new_state.value_avg = max(0, value - config.FpuValue * math.sqrt(total_expanded_policy)) # Lower bound of 0

                        # In the paper it sets FpuValue to 0 at the root node when dirichlet noise is enabled
                        # However, at the root node the policy total is always 0 for my mcts so that's reduntant

                    tree.create_node(data=new_state, parent=node.identifier)
                
                # pn = [p/pn_s for p in pn]
                # ps = [p/sum(ps) for p in ps]
                # fig, axs = plt.subplots(2)
                # fig.suptitle('Policy before and after softmax')
                # axs[0].plot(pn)
                # axs[1].plot(ps)
                # plt.savefig(f"{directory_path}/softmax_policy_{MODEL_VERSION}.png")
                # print("saved")
        
        # Node is terminal
        # Update weights based on winner
        # Range values from 0 to 1 so that value avg will be 0.5
        else: 
            winner = node_state.game.winner
            if winner == node_state.game.turn: # If active player wins
                value = 1
            elif winner == 1 - node_state.game.turn: # If opponent wins
                value = (-1 if config.use_tanh else 0)
            else: # Draw
                value = (0 if config.use_tanh else 0.5)

        # If root node and in self play, add exploration noise to children
        if (config.training and not fast_iter and config.use_dirichlet_noise and node.is_root()):
            child_ids = node.successors(tree.identifier)
            number_of_children = len(child_ids)
            d_alpha = config.DIRICHLET_ALPHA
            if config.use_dirichlet_s:
                d_alpha *= config.DIRICHLET_S / number_of_children

            noise_distribution = np.random.gamma(d_alpha, 1, number_of_children)

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
        value = (-value if config.use_tanh else 1 - value)

        # Go back up the tree and updates nodes
        # Propogate positive values for the player made the move, and negative for the other player
        final_node_turn = node_state.game.turn

        while not node.is_root():
            node_state = node.data
            node_state.visit_count += 1
            # Revert value if the other player just went
            node_state.value_sum += (value if node_state.game.turn == final_node_turn else (-value if config.use_tanh else 1 - value))
            node_state.value_avg = node_state.value_sum / node_state.visit_count

            upwards_id = node.predecessor(tree.identifier)
            node = tree.get_node(upwards_id)

        # Repeat for root node
        node_state = node.data
        node_state.visit_count += 1
        node_state.value_sum += (value if node_state.game.turn == final_node_turn else (-value if config.use_tanh else 1 - value))
        node_state.value_avg = node_state.value_sum / node_state.visit_count

    # print(MAX_DEPTH, total_branch//number_branch)

    # Choose a move based on the number of visits
    root = tree.get_node("root")
    root_children_id = root.successors(tree.identifier)
    max_n = 0
    max_id = None

    root_child_n_list = []
    root_child_policy_list = []

    for root_child_id in root_children_id:
        root_child = tree.get_node(root_child_id)
        root_child_n = root_child.data.visit_count
        root_child_n_list.append(root_child_n)
        root_child_policy_list.append(root_child.data.policy)

        if root_child_n >= max_n: # It's possible n is 0 if there are no possible moves
            max_n = root_child_n
            max_id = root_child.identifier

    # fig, axs = plt.subplots(2)
    # fig.suptitle('N Compared to policy')
    # # axs[0].plot(post_noise_policy)
    # axs[0].plot(root_child_policy_list)
    # axs[1].plot(root_child_n_list)
    # plt.savefig(f"{directory_path}/root_n_{config.MAX_ITER}_depth_{MODEL_VERSION}.png")
    # print("saved")

    data = tree.get_node(max_id).data
    move = data.move

    # Prune policie
    if config.use_forced_playouts_and_policy_target_pruning and config.training and not fast_iter:
        post_prune_n_list = []

        most_playouts_child = tree.get_node(max_id)
        most_playouts_CPUCT = most_playouts_child.data.value_avg + config.CPUCT * most_playouts_child.data.policy * math.sqrt(root.data.visit_count) / (1 + most_playouts_child.data.visit_count)
        
        for root_child_id in root_children_id:
            if root_child_id != max_id:
                root_child = tree.get_node(root_child_id)
                if root_child.data.visit_count > 0:
                    # Calculate n_forced_playouts
                    root_child_n_forced = math.sqrt(config.CForcedPlayout * root_child.data.policy * root.data.visit_count)
                    count = 0

                    while True:
                        if root_child.data.visit_count == 1:
                            # Prune children with one playout
                            root_child.data.visit_count = 0
                            break

                        # Subtract up to n playouts so that the CPUCT value is smaller than the max playout CPUCT
                        root_child_CPUCT_minus = root_child.data.value_avg + config.CPUCT * root_child.data.policy * math.sqrt(root.data.visit_count) / (root.data.visit_count) # No +1

                        if count < root_child_n_forced and root_child_CPUCT_minus < most_playouts_CPUCT:
                            count += 1
                            root_child.data.visit_count -= 1
                        
                        else: break
        
            post_prune_n_list.append(tree.get_node(root_child_id).data.visit_count)

    # If the move was fast, don't save
    save_move = not fast_iter

    return move, tree, save_move

def pick_random_move_by_policy(tree: treelib.Tree) -> tuple:
    # Sample a random move from the root node of the tree using the policy as probabilities
    moves, policies = [], []

    root = tree.get_node("root")

    root_children_id = root.successors(tree.identifier)

    for root_child_id in root_children_id:
        root_child_data = tree.get_node(root_child_id).data

        moves.append(root_child_data.move)
        policies.append(root_child_data.policy)

    move = random.choices(moves, policies)[0]
    return move

def get_move_matrix(player, algo=None):
    # Returns a list of all possible moves that a player can make
    # Very Important: With my coordinate system, pieces can be placed with negative x values
    # To avoid negative indexing, the list of moves is shifted by 2
    # It encodes moves from -2 to 8 as indices 0 to 10
    #
    # Algorithm names:
        # 'brute-force': Slow but finds every move
        # 'faster-but-loss' Faster but misses ~2% of moves
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

    # Convert to new policy format (19, ROWS, COLS)
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

            # Convert queue of placements to the policy grid
            for x, y, o in place_location_queue:
                rotation_index = o % len(policy_pieces[piece.type])
                policy_index = policy_piece_to_index[piece.type][rotation_index]

                new_col = x
                new_row = y

                # For these pieces, only keep track of rotations 0 and 3 so the policy is bounded between -1 and 8
                if piece.type in ["Z", "S", "I"]:
                    # For those pieces, rotation 2 is the same as rotation 0
                    # but moved one down
                    if o == 2:
                        new_row += 1
                    # For those pieces, rotation 1 is the same as rotation 3
                    # but moved one to the right
                    if o == 1:
                        new_col += 1

                new_policy[policy_index][new_row][new_col + BUFFER] = 1 # Account for buffer        
        
    return new_policy

def get_move_list(move_matrix, policy_matrix):
    # Returns list of possible moves with their policy
    # Removes buffer
    mask = np.multiply(move_matrix, policy_matrix)

    move_list = np.argwhere(mask != 0)

    # Formats moves from (policy index, row, col) to (value, (policy index, col - BUFFER, row))
    move_list = [(mask[move[0]][move[1]][move[2]], (move[0], move[2] - BUFFER, move[1])) for move in move_list]

    return move_list

def convert_old_policy_to_new_policy(old_data):
    # Old policy had 2 buffer
    # New policy has 1 buffer, and SZI pieces are saved in rotations 0 and 3 instead of 0 and 1

    # Horrible code but it doesn't raise any errors so hopefully it works
    new_data = []
    for set in old_data:
        new_set = []
        for move in set:
            new_move = move
            policy = np.array(new_move.pop(-1))

            new_policy = np.zeros(POLICY_SHAPE)

            # For I, Z, and S pieces, rotation was changed from 1 to 3
            # So for rotation 1, move it one to the right to make it rotation 3
            policy_entries = np.argwhere(policy != 0)
            for idx, row, col in policy_entries:
                piece, rotation = policy_index_to_piece[idx]
                if piece in ["I", "Z", "S"] and rotation == 3:
                    new_policy[idx][row][col - 1 + 1] = policy[idx][row][col]
                else:
                    new_policy[idx][row][col - 1] = policy[idx][row][col]
            
            new_policy = new_policy.tolist()
            new_move.append(new_policy)
            new_set.append(new_move)
        new_data.append(new_set)
    return new_data

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

# Further tests moved to time_move_matrix

##### Neural Network #####
# 
# Player orientation: Active player, other player
# y:
#   Policy: (19 x 25 x 11) = 5225 (Hold x (Rows - 1) x (Columns + 1) x Rotations)
#   Value: (1)

def instantiate_network(config: Config, nn_generator=gen_alphasame_nn, show_summary=True, save_network=True, plot_model=False):
    # Creates a network with random weights
    # 1: For each grid, apply the same neural network, and then use 1x1 kernel and concatenate
    # 1 -> 2: For opponent grid, apply fully connected layer
    # Concatenate active player's kernels/features with opponent's dense layer and non-player specific features
    # Apply value head and policy head 

    model = nn_generator(config)

    if config.model == 'keras':
        if plot_model == True:
            keras.utils.plot_model(model, to_file=f"{directory_path}/model_{MODEL_VERSION}_img.png", show_shapes=True)

        # Loss is the sum of MSE of values and Cross entropy of policies
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate), loss=["mean_squared_error", "categorical_crossentropy"], loss_weights=config.loss_weights)

        if show_summary: model.summary()

        if save_network:
            path = f"{directory_path}/models/{MODEL_VERSION}"
            os.makedirs(path, exist_ok=True)
            model.save(f"{path}/0.keras")

        return model
    elif config.model == 'pytorch':
        if show_summary: print(model)

        if save_network:
            path = f"{directory_path}/pytorch_models/{MODEL_VERSION}"
            os.makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), f"{path}/0")

def train_network_keras(config, model, set):
    # Keras has 2GB limit (?)

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

    # Adjust learning rate HOW???
    ######### K.set_value(model.optimizer.learning_rate, config.learning_rate)

    model.fit(x=features, y=[values, policies], batch_size=64, epochs=config.epochs, shuffle=config.shuffle)

def train_network_pytorch(config, model, set):
    features = list(map(list, zip(*set)))

    for i in range(len(features)):
        features[i] = torch.tensor(features[i])

        if features[i].dim() == 3 and features[i].shape[1] == 26 and features[i].shape[2] == 10:
            # Add channel for grids
            features[i] = features[i].unsqueeze(dim=1)
            # Convert to float
            features[i] = features[i].type(torch.float)
        
    dataset = torch.utils.data.TensorDataset(*features)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)

    loss_fn_1 = nn.MSELoss()
    loss_fn_2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for data_point in dataloader:
        for feat in data_point:
            feat = feat.to(device)

        y_policy = data_point.pop()
        y_value = data_point.pop().type(torch.float)

        # Compute prediction error
        pred_value, pred_policy = model(*data_point)

        # Reshape policy and value
        pred_value = torch.reshape(pred_value, (-1,))
        pred_policy = torch.reshape(pred_policy, (-1, *POLICY_SHAPE))

        loss_1 = loss_fn_1(pred_value, y_value)
        loss_2 = loss_fn_2(pred_policy, y_policy)

        loss = loss_1 + loss_2

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    loss = loss.item()
    print(f"loss: {loss:>7f}")

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

    # Both value and policies are returned as arrays
    policies = policies.reshape(POLICY_SHAPE)
    
    return value[0][0], policies

def evaluate_pytorch(game, model):
    # Use a neural network to return value and policy.
    data = game_to_X(game)
    X = []

    with torch.no_grad():
        for feature in data:
            tensor = torch.tensor(feature)
            if tensor.size() == (ROWS, COLS):
                # Add channel for grids
                tensor = tensor.unsqueeze(dim=0)
                tensor = tensor.type(torch.float)
            
            # Add one dim
            tensor = tensor.unsqueeze(dim=0)

            tensor = tensor.to(device)
            X.append(tensor)
        
        value, policies = model.forward(*X)

    return value.item(), policies.numpy().reshape(POLICY_SHAPE)

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
            probability_matrix[policy_index][row][col + BUFFER] = round(root_child_n / total_n, 4)

    return probability_matrix


def simplify_grid(grid):
    # Replaces minos in a grid with 1s.
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != 0:
                grid[row][col] = 1
    return grid

# Helper function to reverse data if needed
def reverse_if_needed(data, condition):
    return data[::-1] if condition else data

# Methods for getting game data
# All of them orient the info in the perspective of the active player
def get_grids(game):
    grids = [[x[:] for x in player.board.grid] for player in game.players]
    for grid in grids:
        simplify_grid(grid)
    return reverse_if_needed(grids, game.turn == 1)

def get_pieces(game):
    piece_table = np.zeros((2, 2 + PREVIEWS, len(MINOS)), dtype=int)
    for i, player in enumerate(game.players):
        if player.piece:  # Active piece: 0
            piece_table[i][0][MINOS.index(player.piece.type)] = 1
        if player.held_piece:  # Held piece: 1
            piece_table[i][1][MINOS.index(player.held_piece)] = 1
        # Limit previews
        for j, piece in enumerate(player.queue.pieces[:PREVIEWS]):  # Queue pieces: 2-6
            piece_table[i][j + 2][MINOS.index(piece)] = 1
    return reverse_if_needed(piece_table, game.turn == 1)

def get_stat(game, stat_name):
    stats = [getattr(player.stats, stat_name) for player in game.players]
    return reverse_if_needed(stats, game.turn == 1)

def game_to_X(game):
    # Returns game information for the network.
    # Orient all info in perspective to the current player
    grids = get_grids(game)
    pieces = get_pieces(game)
    b2b = get_stat(game, 'b2b')
    combo = get_stat(game, 'combo')
    lines_cleared = get_stat(game, 'lines_cleared')
    lines_sent = get_stat(game, 'lines_sent')
    color = game.players[game.turn].color
    pieces_placed = game.players[game.turn].stats.pieces

    return (
        grids[0], pieces[0], b2b[0], combo[0], lines_cleared[0], lines_sent[0], 
        grids[1], pieces[1], b2b[1], combo[1], lines_cleared[1], lines_sent[1], 
        color, pieces_placed
    )

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
        # Z, S, and I have rotations 0 and 3

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
            # If the new rotation is 1, it needs to be shifted back after being flipped
            if new_rotation == 1:
                post_col_adjustment = 1

        new_policy_index = policy_piece_to_index[new_piece][new_rotation] # This uses 1 instead of 3 for ZSI index
        for col in range(POLICY_SHAPE[2]):
            for row in range(POLICY_SHAPE[1]):
                value = policy_matrix[policy_index][row][col]
                if value > 0:
                    new_row = row
                    new_col = col

                    # Remove buffer
                    new_col -= BUFFER

                    # Flip column
                    new_col = 10 - new_col - piece_size # 9 - col - piece_size + 1
                    
                    # Add back buffer
                    new_col += BUFFER

                    # Add column adjustment if needed
                    new_col += post_col_adjustment

                    reflected_policy_matrix[new_policy_index][new_row][new_col] = value
    
    return reflected_policy_matrix

def play_game(config, network, game_number=None, show_game=False, screen=None):
    # AI plays one game against itself
    # Returns the game data
    if show_game == True:
        if screen == None:
            screen = pygame.display.set_mode( (WIDTH, HEIGHT))
        pygame.display.set_caption(f'Training game {game_number}')

        for event in pygame.event.get():
            pass

    game = Game()
    game.setup()

    # Initialize data storage
    # Each player's move data will be stored in their respective list
    game_data = [[], []]

    # Initialize the game with random moves
    if config.use_experimental_features:
        scale = 0.04 * COLS * ROWS
        num_random_moves = np.random.exponential(scale=scale)

        fast_config = copy.deepcopy(config)
        fast_config.MAX_ITER = 1
        fast_config.use_playout_cap_randomization = False
        fast_config.use_dirichlet_noise = False
        fast_config.use_forced_playouts_and_policy_target_pruning = False

        while num_random_moves > 0 and game.is_terminal == False:
            # Play random moves proportional to the raw policy distribution
            _, temp_tree, _ = MCTS(fast_config, game, network)

            move = pick_random_move_by_policy(temp_tree)
            game.make_move(move)

            num_random_moves -= 1

    while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
        move, tree, save = MCTS(config, game, network)

        if save or config.save_all:
            search_matrix = search_statistics(tree) # Moves that the network looked at
            
            # Get data
            move_data = [*game_to_X(game)] 

            if config.augment_data:
                # Piece sizes are needed to know where a reflected piece ends up
                reflected_search_matrix = reflect_policy(search_matrix)
                
                # Reflect each player for a total of 2 * 2 = 4 times more data
                # When the other player is reflected, it shouldn't impact active player
                # When the active player is reflected, the search placements need to be reflected as well
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
            else:
                # Convert to regular lists
                for i in range(len(move_data)):
                    if isinstance(move_data[i], np.ndarray):
                        move_data[i] = move_data[i].tolist()
                
                move_data.append(search_matrix)
                game_data[game.turn].append(move_data)

        game.make_move(move)

        if show_game == True:
            game.show(screen)
            pygame.display.update()

    # After game ends update value
    winner = game.winner
    for player_idx in range(len(game_data)):
        if winner == -1: # Draw
            value = (0 if config.use_tanh else 0.5)
        elif winner == player_idx:
            value = 1
        else:
            value = (-1 if config.use_tanh else 0)
        # Insert value before policy for each move of that player
        for move_idx in range(len(game_data[player_idx])):
            game_data[player_idx][move_idx].insert(-1, value)

    # Reformat data to stack all moves into one continuous list
    data = game_data[0]
    data.extend(game_data[1])

    return data

def make_training_set(config, network, num_games, save_game=True, show_game=False, screen=None):
    # Creates a dataset of several AI games.
    series_data = []
    for idx in range(1, num_games + 1):
        data = play_game(config, network, game_number=idx, show_game=show_game, screen=screen)
        series_data.extend(data)

    if save_game == True:
        json_data = ujson.dumps(series_data)

        # Increment set counter
        next_set = highest_data_number(DATA_VERSION) + 1

        with open(f"{directory_path}/data/{DATA_VERSION}/{next_set}.txt", 'w') as out_file:
            out_file.write(json_data)
    
    else:
        return series_data

def load_data(data_ver=DATA_VERSION, last_n_sets=SETS_TO_TRAIN_WITH):
    data = []
    # Load data from the past n games
    # You can input model_ver or model_iter as None to load 
    # all iterations from one version OR load all data files
    # under past last_n

    # Find highest set number
    # SHOULD ONLY BE USED IF model_ver AND model_iter are specified
    max_set = highest_data_number(data_ver)

    sets, len_sets = 0, 0

    path = f"{directory_path}/data/{data_ver}"

    # Get n games
    for filename in os.listdir(path):
        data_number = int(filename.split('.')[0])
        if data_number > max_set - last_n_sets:
            # Load data
            set = ujson.load(open(f"{path}/{filename}", 'r'))
            data.append(set)

            sets += 1
            len_sets += len(set)
    
    print(sets, len_sets)
    # Shuffle data
    random.shuffle(data)
    return data

def get_data_filenames(data_ver=DATA_VERSION, last_n_sets=SETS_TO_TRAIN_WITH, shuffle=True):
    filenames = []
    max_set = highest_data_number(data_ver)

    sets = 0

    path = f"{directory_path}/data/{data_ver}"

    # Get n games
    for filename in os.listdir(path):
        data_number = int(filename.split('.')[0])
        if data_number > max_set - last_n_sets:
            # Load data
            filenames.append(filename)

            sets += 1
    
    print(sets)
    # Shuffle data
    if shuffle:
        random.shuffle(filenames)
    return filenames
    

def battle_networks(NN_1, config_1, NN_2, config_2, threshold, threshold_type, games, network_1_title='Network 1', network_2_title='Network 2', show_game=False, screen=None):
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
                move, _, _ = MCTS(config_1, game, NN_1)    
            elif game.turn == 1:
                move, _, _ = MCTS(config_2, game, NN_2)
            game.make_move(move)

            if show_game == True:
                game.show(screen)
                pygame.display.update()

        winner = game.winner
        if winner == -1:
            wins += 0.5
        else: wins[winner] += 1

        # End early if either player reaches the cutoff
        if threshold_type == 'more':
            if wins[0] > threshold * games:
                print(*wins)
                return True
            elif wins[1] >= (1 - threshold) * games:
                print(*wins)
                return False
        elif threshold_type == 'moreorequal':
            if wins[0] >= threshold * games:
                print(*wins)
                return True
            elif wins[1] > (1 - threshold) * games:
                print(*wins)
                return False
        else:
            raise NotImplementedError

    # If neither side eaches a cutoff (which shouldn't happen) return false
    return False

def self_play_loop(config, skip_first_set=False, show_games=False):
    if show_games == True:
        screen = pygame.display.set_mode( (WIDTH, HEIGHT))
    # Given a network, generates training data, trains it, and checks if it improved.
    best_network = load_best_model(config)

    if config.model == 'keras':
        best_interpreter = get_interpreter(best_network)

    training_config = copy.deepcopy(config)
    # Setting training to true enables a variety of training features
    training_config.training = True

    iter = 0

    while True:
        iter += 1

        # The challenger network will be trained and then battled against the prior network
        challenger_network = load_best_model(config)

        # Play a training set and train the network on past sets.
        if not skip_first_set:
            for i in range(TRAINING_LOOPS):
                # Make data file with trianing config
                make_training_set(training_config, (best_interpreter if config.model == 'keras' else best_network), TRAINING_GAMES, show_game=show_games, screen=screen)
                print("Finished set")
        else:
            # Skip and stop skipping in future
            skip_first_set = False

        # Load data
        filenames = get_data_filenames(last_n_sets=SETS_TO_TRAIN_WITH, shuffle=config.shuffle)

        path = f"{directory_path}/data/{DATA_VERSION}"

        for filename in filenames:
            set = ujson.load(open(f"{path}/{filename}", 'r'))

            # Train challenger network
            if config.model == 'keras':
                train_network_keras(config, challenger_network, set)
            elif config.model == 'pytorch':
                train_network_pytorch(config, challenger_network, set)

        del set

        if config.model == 'keras':
            challenger_interpreter = get_interpreter(challenger_network)

        print("Finished loop")

        # If new network is improved, save it and make it the default
        # Otherwise, repeat
        if battle_networks(
            (challenger_interpreter if config.model == 'keras' else challenger_network), 
            config, 
            (best_interpreter if config.model == 'keras' else best_network), 
            config, 
            GATING_THRESHOLD, 
            GATING_THRESHOLD_TYPE,
            BATTLE_GAMES, 
            show_game=show_games, 
            screen=screen
        ):
            # Challenger network becomes next highest version
            next_ver = highest_model_number(config, MODEL_VERSION) + 1

            if config.model == 'keras':
                challenger_network.save(f"{directory_path}/models/{MODEL_VERSION}/{next_ver}.keras")
            if config.model == 'pytorch':
                torch.save(challenger_network.state_dict(), f"{directory_path}/pytorch_models/{MODEL_VERSION}/{next_ver}")

            # The new network becomes the network to beat
            best_network = challenger_network

            if config.model == 'keras':
                best_interpreter = get_interpreter(best_network)
        
        del challenger_network
        if config.model == 'keras':
            del challenger_interpreter

        gc.collect()

def load_best_model(config):
    max_ver = highest_model_number(config, MODEL_VERSION)

    if config.model == 'keras':
        path = f"{directory_path}/models/{MODEL_VERSION}/{max_ver}.keras"

        model = keras.models.load_model(path)
    elif config.model == 'pytorch':
        path = f"{directory_path}/pytorch_models/{MODEL_VERSION}/{max_ver}"

        model = config.default_model(config)
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()

    print(path)

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

def highest_model_number(config, ver):
    max = -1

    if config.model == 'keras':
        path = f"{directory_path}/models/{ver}"

        os.makedirs(path, exist_ok=True)

        for filename in os.listdir(path):
            model_number = int(filename.split('.')[0])
            if model_number > max:
                max = model_number

    elif config.model == 'pytorch':
        path = f"{directory_path}/pytorch_models/{ver}"

        os.makedirs(path, exist_ok=True)

        for filename in os.listdir(path):
            model_number = int(filename)
            if model_number > max:
                max = model_number

    return max

def highest_data_number(data_ver):
    max = -1

    path = f"{directory_path}/data/{data_ver}"

    os.makedirs(path, exist_ok=True)

    for filename in os.listdir(path):
        data_number = int(filename.split('.')[0])
        if data_number > max:
            max = data_number

    return max

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