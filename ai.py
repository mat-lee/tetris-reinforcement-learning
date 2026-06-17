# # Prior to importing tensorflow, disable debug logs
from architectures import *
from const import *
from game import Game
from piece_location import PieceLocation
from move_generation import get_move_matrix

import asyncio
import copy
from collections import deque
from datetime import datetime
import gc
#import json
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
from scipy import signal
import sys
import time
import ujson

from tensorflow import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import math_ops
from sklearn.model_selection import train_test_split

# Reduce tensorflow text
tf.get_logger().setLevel(logging.ERROR)

# Suppress TensorFlow Lite warnings specifically
tf.get_logger().setLevel('ERROR')

from torch.utils.data import DataLoader

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Not sure where I'm importing pygame from but sure
pygame.init()

import cProfile
import pstats

# For reducing the amount of tensorflow prints
HIDE_PRINTS = True

# Where data and models are saved
directory_path = Path.cwd().parent / "Storage"
logs_path = directory_path / "logs"
logs_path.mkdir(exist_ok=True)


class Config():
    def __init__(
        self, 

        visual=True, # Whether to display the training

        # For naming data and models
        model_version=7.0,
        data_version=2.9,

        ruleset='s2', # 's1' for season 1, 's2' for season 2

        model='keras', # 'keras' or 'pytorch'
        use_tflite=True, # If true uses tflite, otherwise uses keras directly. Only for keras models
                         # If uses tflite, then interference and training are separate classes
                         # Otherwise, interference and training are the same class
        tflite_num_threads=2, # CPU threads for the tflite interpreter. Default 2 keeps the
                              # laptop responsive; raise to use more cores during long batch jobs.
        batched_inference=False, # Pytorch only. If True, self-play and battle run all games
                                # concurrently and coalesce NN calls into one batched forward.
                                # If False, falls back to the original serial loop (one game at a time, BS=1).
        model_config=AuxBaseResNetConfig(), # Architecture Parameters
        move_algorithm='convolutional', # 'brute-force' for brute force, 'faster-but-loss' for faster but less accurate, 'harddrop' for harddrops only

        use_tanh=False, # If false means using sigmoid; affects data saving and model activation
        # Makes evaluation range from -1 to 1, while sigmoid ranges from 0 to 1

        # MCTS Parameters
        training_games=100, # Number of training games per training loop
        training_loops=1, # Number of training loops before evaluation
        sets_to_train_with=10, # Number of past sets to train with
        battle_games=200, # Number of evaluation games
        gating_threshold=0.52, # Minimum winrate to replace the best model
        gating_threshold_type='moreorequal', # 'moreorequal' or 'more'

        MAX_ITER=400, 
        CPUCT=0.75, # CPUCT is the scalar multiple of the policy term in PUCT
        DPUCT=1, # DPUCT is an additive scalar in the denominator of in PUCT

        FpuStrategy='reduction', # 'reduction' subtracts FpuValue from parent eval, 'absolute' uses FpuValue
        FpuValue=0.1,

        use_root_softmax=True,
        RootSoftmaxTemp=1.1,

        temperature=0.1,

        # Training Parameters
        training=False, # Set to true to use a variety of features
        learning_rate=0.001,
        weight_decay=0.0,
        epochs=1,
        batch_size=64,

        data_loading_style='merge', # 'merge' combines sets for training, 'distinct' trains across sets first
        decay_factor=0.9, # Only for 'merge' data loading style and when data is None
        augment_data=True,
        shuffle=True,
        use_experimental_features=False, # Before setting to true, check if it's in use
        save_all=False,
        loss_weights=[1, 1],

        use_random_starting_moves=False, # If true, pick the first few moves randomly with respect to policy weights

        use_playout_cap_randomization=True,
        playout_cap_chance=0.25,
        playout_cap_mult=5,

        use_dirichlet_noise=True,
        DIRICHLET_ALPHA=0.1,
        DIRICHLET_S=25,
        DIRICHLET_EXPLORATION=0.25, 
        use_dirichlet_s=True,

        use_forced_playouts_and_policy_target_pruning=False,
        CForcedPlayout=1,
    ):
        self.visual = visual
        self.model_version = model_version
        self.data_version = data_version
        self.ruleset = ruleset
        self.model = model
        self.use_tflite = use_tflite
        self.tflite_num_threads = tflite_num_threads
        self.batched_inference = batched_inference
        self.model_config = model_config
        self.move_algorithm = move_algorithm
        self.use_tanh = use_tanh
        self.training_games = training_games
        self.training_loops = training_loops
        self.sets_to_train_with = sets_to_train_with
        self.battle_games = battle_games
        self.gating_threshold = gating_threshold
        self.gating_threshold_type = gating_threshold_type
        self.MAX_ITER = MAX_ITER
        self.CPUCT = CPUCT
        self.DPUCT = DPUCT
        self.FpuStrategy = FpuStrategy
        self.FpuValue = FpuValue
        self.use_root_softmax = use_root_softmax
        self.RootSoftmaxTemp = RootSoftmaxTemp
        self.temperature = temperature
        self.training = training
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_loading_style = data_loading_style
        self.decay_factor = decay_factor
        self.augment_data = augment_data
        self.shuffle = shuffle
        self.use_experimental_features = use_experimental_features
        self.save_all = save_all
        self.loss_weights = loss_weights
        self.use_random_starting_moves = use_random_starting_moves
        self.use_playout_cap_randomization = use_playout_cap_randomization
        self.playout_cap_chance = playout_cap_chance
        self.playout_cap_mult = playout_cap_mult
        self.use_dirichlet_noise = use_dirichlet_noise
        self.DIRICHLET_ALPHA = DIRICHLET_ALPHA
        self.DIRICHLET_S = DIRICHLET_S
        self.DIRICHLET_EXPLORATION = DIRICHLET_EXPLORATION
        self.use_dirichlet_s = use_dirichlet_s
        self.use_forced_playouts_and_policy_target_pruning = use_forced_playouts_and_policy_target_pruning
        self.CForcedPlayout = CForcedPlayout

    def copy(self):
        # Create a new Config instance with the same attributes as self
        return Config(**vars(self))

    @property
    def model_dir(self):
        if self.model == 'pytorch':
            return f"{directory_path}/pytorch_models/{self.ruleset}.{self.model_version}"
        return f"{directory_path}/models/{self.ruleset}.{self.model_version}"
    
    @property
    def data_dir(self):
        # Returns the path to the data file
        return f"{directory_path}/data/{self.ruleset}.{self.data_version}"

    @property
    def value_max(self):
        return 1

    @property
    def value_mid(self):
        return (0 if self.use_tanh else 0.5)

    @property
    def value_min(self):
        return (-1 if self.use_tanh else 0)
    
    def negate_value(self, value):
        return (-value if self.use_tanh else 1 - value)


def config_to_dict(config):
    d = {k: v for k, v in vars(config).items() if k != 'model_config'}
    d['model_config'] = vars(config.model_config)
    return d

def append_version_record(config, model_number):
    path = Path(config.model_dir) / "versions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_number": model_number,
        "backend": config.model,
        "config": config_to_dict(config),
    }
    with open(path, 'a') as f:
        f.write(ujson.dumps(entry) + '\n')


class MCTSNode:
    """Lightweight MCTS tree node. Replaces treelib.Node to avoid UUID overhead."""
    __slots__ = ('identifier', 'data', '_parent_id', '_children')

    def __init__(self, identifier, data, parent_id=None):
        self.identifier = identifier
        self.data = data
        self._parent_id = parent_id
        self._children = []

    def is_root(self):
        return self._parent_id is None

    def is_leaf(self):
        return len(self._children) == 0

    def successors(self, _tree_id=None):
        return self._children

    def predecessor(self, _tree_id=None):
        return self._parent_id


class MCTSTree:
    """Lightweight MCTS tree backed by a plain dict. Replaces treelib.Tree."""

    def __init__(self):
        self._nodes = {}
        self._counter = 0
        self.identifier = 0  # kept for API compatibility (successors/predecessor pass this)

    def create_node(self, identifier=None, data=None, parent=None):
        if identifier is None:
            self._counter += 1
            identifier = self._counter
        node = MCTSNode(identifier=identifier, data=data, parent_id=parent)
        self._nodes[identifier] = node
        if parent is not None:
            self._nodes[parent]._children.append(identifier)
        return node

    def get_node(self, identifier):
        return self._nodes[identifier]


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

def MCTS(config, game, interference_network) -> tuple[tuple, MCTSTree, bool]:
    # Picks a move for the AI to make 

    # Initialize the search tree
    tree = MCTSTree()
    game_copy = game.copy()

    # Restrict previews
    for player in game_copy.players:
        while len(player.queue.pieces) > PREVIEWS:
            player.queue.pieces.pop(-1)

    # Create the root node
    initial_state = NodeState(game=game_copy, move=None)

    tree.create_node(identifier="root", data=initial_state)

    MAX_DEPTH = 0
    iter = 0

    max_iterations = None
    fast_iter = False # Used to disable dirichlet noise for fast iterations

    # Randomly assigns the MCTS to a higher or lower playouts
    if config.training and config.use_playout_cap_randomization:
        if random.random() < config.playout_cap_chance: # long
            max_iterations = math.ceil(config.playout_cap_mult * (config.MAX_ITER / (config.playout_cap_chance * (config.playout_cap_mult - 1) + 1)))
        else: # short
            max_iterations = math.floor(config.MAX_ITER / (config.playout_cap_chance * (config.playout_cap_mult - 1) + 1))
            fast_iter = True
    else: # default
        max_iterations = config.MAX_ITER

    while iter < max_iterations:
        iter += 1

        # Begin at the root node
        node = tree.get_node("root")
        node_state = node.data

        DEPTH = 0

        if iter == config.MAX_ITER - 1:
            pass

        # Go down the tree using formula Q+U until you get to a leaf node
        # However, if using forced playouts, select a node if it has fewer than the forced playouts amount
        while not node.is_leaf():
            child_ids = node.successors(tree.identifier)
            max_child_score = -1
            max_child_id = None
            parent_visits = node.data.visit_count

            sqrt_parent = math.sqrt(parent_visits)
            unvisited_U_scale = config.CPUCT * sqrt_parent / config.DPUCT
            check_forced = (
                config.use_forced_playouts_and_policy_target_pruning
                and config.training
                and node.is_root()
                and not (config.use_playout_cap_randomization and fast_iter)
            )

            # Look through each child
            for child_id in child_ids:

                # For each child calculate a score
                # Polynomial upper confidence trees (PUCT)
                child_data = tree.get_node(child_id).data

                Q = child_data.value_avg
                vc = child_data.visit_count
                if vc == 0:
                    U = unvisited_U_scale * child_data.policy
                else:
                    U = config.CPUCT * child_data.policy * sqrt_parent / (config.DPUCT + vc)

                child_score = Q + U

                # Check forced playouts
                if check_forced and vc >= 1:
                    n_forced = math.sqrt(config.CForcedPlayout * child_data.policy * parent_visits)
                    if vc < n_forced:
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

        playout_node_id = node.identifier

        # If not the root node, place piece in node
        if not node.is_root():
            prior_node = tree.get_node(node.predecessor(tree.identifier))
            
            game_copy = prior_node.data.game.copy()
            node_state.game = game_copy
            node_state.game.make_move(node_state.move, add_bag=False, add_history=False)

        # Update policy, move_list and generate new nodes
        if node_state.game.is_terminal == False: # Avoid is node game is over
            value, policy = evaluate(config, node_state.game, interference_network)
            # value, policy = random_evaluate()
                
            # Make sure that no values of the policy are below 0
            policy[policy<=0] = 1e-25

            if node_state.game.no_move == False:
                move_matrix = get_move_matrix(node_state.game.players[node_state.game.turn], algo=config.move_algorithm)
                move_list = get_move_list(move_matrix, policy)

                assert len(move_list) > 0 # There should always be a legal move

                # Calculate policy for new leaves
                policies, moves = map(list, zip(*move_list))

                # Debugging
                # pn = []
                # ps = []
                # pn_s = sum(policies)

                # Apply softmax and temperature to policy of children of the root node
                if node.is_root() and config.use_root_softmax:
                    # Formula taken from katago
                    max_policy = max(policies)
                    log_max = math.log(max_policy)
                    inv_temp = 1.0 / config.RootSoftmaxTemp
                    for i in range(len(policies)):
                        policies[i] = math.exp((math.log(policies[i]) - log_max) * inv_temp)

                policy_sum = sum(policies)

                # Generate leaf nodes
                for policy, move in zip(policies, moves):
                    new_state = NodeState(game=None, move=move)

                    # New node policy
                    new_state.policy = policy / policy_sum # Normalize
                    assert new_state.policy > 0

                    # New node value
                    if config.FpuStrategy == 'absolute':
                        new_state.value_avg = max(config.value_min, config.FpuValue)
                    elif config.FpuStrategy == 'reduction':
                        # Total explored policy for new leaf node is 0
                        # Flip the perspective
                        parent_value = config.negate_value(value)
                        new_state.value_avg = max(config.value_min, parent_value)

                        # In the paper it sets FpuValue to 0 at the root node when dirichlet noise is enabled
                        # However, at the root node the policy total is always 0 for my mcts so that's reduntant

                    tree.create_node(data=new_state, parent=node.identifier)
                
                # pn = [p/pn_s for p in pn]
                # ps = [p/sum(ps) for p in ps]
                # fig, axs = plt.subplots(2)
                # fig.suptitle('Policy before and after softmax')
                # axs[0].plot(pn)
                # axs[1].plot(ps)
                # plt.savefig(f"{directory_path}/softmax_policy_{config.model_version}.png")
                # print("saved")
        
        # Node is terminal
        # Update weights based on winner
        # Range values from 0 to 1 so that value avg will be 0.5
        else: 
            winner = node_state.game.winner
            if winner == node_state.game.turn: # If active player wins
                value = config.value_max
            elif winner == 1 - node_state.game.turn: # If opponent wins
                value = config.value_min
            else: # Draw
                value = config.value_mid

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
        value = config.negate_value(value)
        pos_value = value
        neg_value = config.negate_value(value)

        # Go back up the tree and updates nodes
        # Propogate positive values for the player made the move, and negative for the other player
        final_node_turn = node_state.game.turn

        while not node.is_root():
            node_state = node.data
            node_state.visit_count += 1
            # Revert value if the other player just went
            node_state.value_sum += (pos_value if node_state.game.turn == final_node_turn else neg_value)
            node_state.value_avg = node_state.value_sum / node_state.visit_count

            upwards_id = node.predecessor(tree.identifier)
            node = tree.get_node(upwards_id)

        # Repeat for root node
        node_state = node.data
        node_state.visit_count += 1
        node_state.value_sum += (pos_value if node_state.game.turn == final_node_turn else neg_value)
        node_state.value_avg = node_state.value_sum / node_state.visit_count


        # After playouts are updated, update Fpu values

        # Update Fpu values if FpuStragegy is 'reduction'
        # If the parent is the root, no updates will occur
        # The var 'node' is the node that was just played out
        # Go back to the parent, and update all its children's ("node"`s siblings) fpu
        node = tree.get_node(playout_node_id)

        if not node.is_root() and config.FpuStrategy == 'reduction':
            parent_id = node.predecessor(tree.identifier)
            parent = tree.get_node(parent_id)

            if not parent.is_root():
                parent_value = parent.data.value_avg

                node_explored_policy = 0
                unvisited_states = []

                # Single pass: collect explored policy and unvisited siblings
                sibling_ids = parent.successors(tree.identifier)
                for sibling_id in sibling_ids:
                    sibling_data = tree.get_node(sibling_id).data
                    if sibling_data.visit_count > 0:
                        node_explored_policy += sibling_data.policy
                    else:
                        unvisited_states.append(sibling_data)

                fpu = max(config.value_min, config.negate_value(parent_value) - config.FpuValue * math.sqrt(node_explored_policy))
                for sibling_data in unvisited_states:
                    sibling_data.value_avg = fpu


    # ----- Pick a move randomly using temperature BEFORE pruning visit counts -----

    # Find the move with the highest number of playouts
    root = tree.get_node("root")
    root_children_id = root.successors(tree.identifier)
    max_n = 0
    max_id = None

    root_child_n_list = []
    root_child_id_list = []

    root_child_policy_list = [] # debugging

    for root_child_id in root_children_id:
        root_child = tree.get_node(root_child_id)
        root_child_n = root_child.data.visit_count
        root_child_n_list.append(root_child_n)
        root_child_id_list.append(root_child_id)

        if root_child_n >= max_n: # It's possible n is 0 if there are no possible moves
            max_n = root_child_n
            max_id = root_child.identifier
        
        root_child_policy_list.append(root_child.data.policy) # debugging

    selected_id = None
    
    def select_action_with_temperature(visit_counts, temperature):
        if temperature == 0:
            # Deterministic - pick most visited
            return np.argmax(visit_counts)
        else:
            # Stochastic - higher temp = more random
            probs = visit_counts ** (1 / temperature)
            probs = probs / np.sum(probs)

            idx = random.choices(range(len(probs)), weights=probs, k=1)[0]
            return idx
    
    # Use post pruned n if pruning occurred
    # Set temperature to 0 if not training
    temp = config.temperature if config.training else 0

    selected_idx = select_action_with_temperature(np.array(root_child_n_list), temp)
    selected_id = root_child_id_list[selected_idx]

    move = tree.get_node(selected_id).data.move

    # ----- Prune policy AFTER choosing a move -----
    # Prune policy
    post_prune_n_list = None
    if config.use_forced_playouts_and_policy_target_pruning and config.training and not fast_iter:
        post_prune_n_list = []

        most_playouts_child = tree.get_node(max_id) # Uses max_id, not selected_id
        most_playouts_CPUCT = most_playouts_child.data.value_avg + config.CPUCT * most_playouts_child.data.policy * math.sqrt(root.data.visit_count) / (config.DPUCT + most_playouts_child.data.visit_count)
        
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
                        root_child_CPUCT_minus = root_child.data.value_avg + config.CPUCT * root_child.data.policy * math.sqrt(root.data.visit_count) / (config.DPUCT + root.data.visit_count)

                        if count < root_child_n_forced and root_child_CPUCT_minus < most_playouts_CPUCT:
                            count += 1
                            root_child.data.visit_count -= 1
                        
                        else: break
        
            post_prune_n_list.append(tree.get_node(root_child_id).data.visit_count)

    if post_prune_n_list is not None and False: # debugging forced playout pruning
        pre_prune_num_moves = len([x for x in root_child_n_list if x > 0])
        post_prune_num_moves = len([x for x in post_prune_n_list if x > 0])

        print(f"Move diversity: {pre_prune_num_moves} → {post_prune_num_moves} moves")

    # If the move was fast, don't save
    save_move = not fast_iter

    return move, tree, save_move


# ============================================================================
# Async batched MCTS (PyTorch only).
# Cross-game leaf parallelization: N concurrent games coalesce their NN
# evaluations into one model.forward(batch=K). Each tree is still pure
# sequential PUCT — search quality is identical to the sync MCTS above.
# If you change MCTS(), mirror the change in amcts() below.
# ============================================================================

class BatchedEvaluator:
    """Coalesces concurrent aevaluate() calls into one model.forward(batch=K)."""
    def __init__(self, model, config, max_batch=128, timeout_ms=2):
        self.model = model
        self.config = config
        self.max_batch = max_batch
        self.timeout_s = timeout_ms / 1000
        self._queue = None  # created in start() to bind to running loop
        self._task = None
        self._stop = False

    async def start(self):
        self._queue = asyncio.Queue()
        self._stop = False
        self._task = asyncio.create_task(self._worker())

    async def stop(self):
        self._stop = True
        if self._task is not None:
            await self._task
            self._task = None

    async def submit(self, x_features):
        """x_features: list of per-sample tensors (no batch dim), matching
        the 11 outputs of game_to_X. Returns (value: float, policy: np.ndarray)."""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await self._queue.put((x_features, fut))
        return await fut

    async def _worker(self):
        while True:
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                if self._stop and self._queue.empty():
                    return
                continue

            batch = [first]
            # Opportunistic drain — small timeout so undersized batches still go.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.timeout_s
            while len(batch) < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break

            self._dispatch(batch)

    def _dispatch(self, batch):
        n_features = len(batch[0][0])
        # Stack each feature across the batch dim
        xs = []
        for i in range(n_features):
            stacked = torch.stack([b[0][i] for b in batch], dim=0).to(device)
            xs.append(stacked)

        with torch.no_grad():
            out = self.model.forward(*xs)
        values_t, policies_t = out[0], out[1]

        # Softmax per-sample, matching evaluate_pytorch
        policies_flat = torch.softmax(policies_t.reshape(len(batch), -1), dim=1)
        values_np = values_t.detach().cpu().numpy().reshape(-1)
        policies_np = policies_flat.detach().cpu().numpy().reshape(len(batch), *POLICY_SHAPE)

        for i, (_, fut) in enumerate(batch):
            fut.set_result((float(values_np[i]), policies_np[i]))


def _build_features_for_batcher(game):
    """Build the per-sample feature list (no batch dim) for BatchedEvaluator.submit.
    Mirrors evaluate_pytorch's per-sample tensor construction minus the unsqueeze(0)."""
    data = game_to_X(game)
    X = []
    for feature in data:
        tensor = torch.tensor(feature)
        if tensor.size() == (ROWS, COLS):
            tensor = tensor.unsqueeze(dim=0)  # add channel dim → (1, ROWS, COLS)
            tensor = tensor.type(torch.float)
        X.append(tensor)
    return X


async def aevaluate(config, game, evaluator: BatchedEvaluator):
    """Async drop-in for evaluate(): submits features to the batcher and awaits result."""
    x = _build_features_for_batcher(game)
    value, policy = await evaluator.submit(x)
    return value, policy


async def amcts(config, game, evaluator: BatchedEvaluator) -> tuple[tuple, MCTSTree, bool]:
    """Async mirror of MCTS(). Identical PUCT logic; only evaluate() is awaited."""
    tree = MCTSTree()
    game_copy = game.copy()

    for player in game_copy.players:
        while len(player.queue.pieces) > PREVIEWS:
            player.queue.pieces.pop(-1)

    initial_state = NodeState(game=game_copy, move=None)
    tree.create_node(identifier="root", data=initial_state)

    MAX_DEPTH = 0
    iter = 0

    max_iterations = None
    fast_iter = False

    if config.training and config.use_playout_cap_randomization:
        if random.random() < config.playout_cap_chance:
            max_iterations = math.ceil(config.playout_cap_mult * (config.MAX_ITER / (config.playout_cap_chance * (config.playout_cap_mult - 1) + 1)))
        else:
            max_iterations = math.floor(config.MAX_ITER / (config.playout_cap_chance * (config.playout_cap_mult - 1) + 1))
            fast_iter = True
    else:
        max_iterations = config.MAX_ITER

    while iter < max_iterations:
        iter += 1
        node = tree.get_node("root")
        node_state = node.data
        DEPTH = 0

        while not node.is_leaf():
            child_ids = node.successors(tree.identifier)
            max_child_score = -1
            max_child_id = None
            parent_visits = node.data.visit_count

            sqrt_parent = math.sqrt(parent_visits)
            unvisited_U_scale = config.CPUCT * sqrt_parent / config.DPUCT
            check_forced = (
                config.use_forced_playouts_and_policy_target_pruning
                and config.training
                and node.is_root()
                and not (config.use_playout_cap_randomization and fast_iter)
            )

            for child_id in child_ids:
                child_data = tree.get_node(child_id).data
                Q = child_data.value_avg
                vc = child_data.visit_count
                if vc == 0:
                    U = unvisited_U_scale * child_data.policy
                else:
                    U = config.CPUCT * child_data.policy * sqrt_parent / (config.DPUCT + vc)
                child_score = Q + U

                if check_forced and vc >= 1:
                    n_forced = math.sqrt(config.CForcedPlayout * child_data.policy * parent_visits)
                    if vc < n_forced:
                        child_score = float('inf')

                if child_score >= max_child_score:
                    max_child_score = child_score
                    max_child_id = child_id

            node = tree.get_node(max_child_id)
            node_state = node.data
            DEPTH += 1
            if DEPTH > MAX_DEPTH:
                MAX_DEPTH = DEPTH

        playout_node_id = node.identifier

        if not node.is_root():
            prior_node = tree.get_node(node.predecessor(tree.identifier))
            game_copy = prior_node.data.game.copy()
            node_state.game = game_copy
            node_state.game.make_move(node_state.move, add_bag=False, add_history=False)

        if node_state.game.is_terminal == False:
            value, policy = await aevaluate(config, node_state.game, evaluator)
            policy[policy <= 0] = 1e-25

            if node_state.game.no_move == False:
                move_matrix = get_move_matrix(node_state.game.players[node_state.game.turn], algo=config.move_algorithm)
                move_list = get_move_list(move_matrix, policy)
                assert len(move_list) > 0

                policies, moves = map(list, zip(*move_list))

                if node.is_root() and config.use_root_softmax:
                    max_policy = max(policies)
                    log_max = math.log(max_policy)
                    inv_temp = 1.0 / config.RootSoftmaxTemp
                    for i in range(len(policies)):
                        policies[i] = math.exp((math.log(policies[i]) - log_max) * inv_temp)

                policy_sum = sum(policies)

                for policy_p, move in zip(policies, moves):
                    new_state = NodeState(game=None, move=move)
                    new_state.policy = policy_p / policy_sum
                    assert new_state.policy > 0

                    if config.FpuStrategy == 'absolute':
                        new_state.value_avg = max(config.value_min, config.FpuValue)
                    elif config.FpuStrategy == 'reduction':
                        parent_value = config.negate_value(value)
                        new_state.value_avg = max(config.value_min, parent_value)

                    tree.create_node(data=new_state, parent=node.identifier)
        else:
            winner = node_state.game.winner
            if winner == node_state.game.turn:
                value = config.value_max
            elif winner == 1 - node_state.game.turn:
                value = config.value_min
            else:
                value = config.value_mid

        if (config.training and not fast_iter and config.use_dirichlet_noise and node.is_root()):
            child_ids = node.successors(tree.identifier)
            number_of_children = len(child_ids)
            d_alpha = config.DIRICHLET_ALPHA
            if config.use_dirichlet_s:
                d_alpha *= config.DIRICHLET_S / number_of_children

            noise_distribution = np.random.gamma(d_alpha, 1, number_of_children)

            for child_id, noise in zip(child_ids, noise_distribution):
                child_data = tree.get_node(child_id).data
                child_data.policy = child_data.policy * (1 - config.DIRICHLET_EXPLORATION) + noise * config.DIRICHLET_EXPLORATION

        value = config.negate_value(value)
        pos_value = value
        neg_value = config.negate_value(value)

        final_node_turn = node_state.game.turn

        while not node.is_root():
            node_state = node.data
            node_state.visit_count += 1
            node_state.value_sum += (pos_value if node_state.game.turn == final_node_turn else neg_value)
            node_state.value_avg = node_state.value_sum / node_state.visit_count
            upwards_id = node.predecessor(tree.identifier)
            node = tree.get_node(upwards_id)

        node_state = node.data
        node_state.visit_count += 1
        node_state.value_sum += (pos_value if node_state.game.turn == final_node_turn else neg_value)
        node_state.value_avg = node_state.value_sum / node_state.visit_count

        node = tree.get_node(playout_node_id)

        if not node.is_root() and config.FpuStrategy == 'reduction':
            parent_id = node.predecessor(tree.identifier)
            parent = tree.get_node(parent_id)

            if not parent.is_root():
                parent_value = parent.data.value_avg
                node_explored_policy = 0
                unvisited_states = []
                sibling_ids = parent.successors(tree.identifier)
                for sibling_id in sibling_ids:
                    sibling_data = tree.get_node(sibling_id).data
                    if sibling_data.visit_count > 0:
                        node_explored_policy += sibling_data.policy
                    else:
                        unvisited_states.append(sibling_data)

                fpu = max(config.value_min, config.negate_value(parent_value) - config.FpuValue * math.sqrt(node_explored_policy))
                for sibling_data in unvisited_states:
                    sibling_data.value_avg = fpu

    # Move selection (identical to MCTS)
    root = tree.get_node("root")
    root_children_id = root.successors(tree.identifier)
    max_n = 0
    max_id = None
    root_child_n_list = []
    root_child_id_list = []

    for root_child_id in root_children_id:
        root_child = tree.get_node(root_child_id)
        root_child_n = root_child.data.visit_count
        root_child_n_list.append(root_child_n)
        root_child_id_list.append(root_child_id)
        if root_child_n >= max_n:
            max_n = root_child_n
            max_id = root_child.identifier

    def select_action_with_temperature(visit_counts, temperature):
        if temperature == 0:
            return np.argmax(visit_counts)
        probs = visit_counts ** (1 / temperature)
        probs = probs / np.sum(probs)
        return random.choices(range(len(probs)), weights=probs, k=1)[0]

    temp = config.temperature if config.training else 0
    selected_idx = select_action_with_temperature(np.array(root_child_n_list), temp)
    selected_id = root_child_id_list[selected_idx]
    move = tree.get_node(selected_id).data.move

    post_prune_n_list = None
    if config.use_forced_playouts_and_policy_target_pruning and config.training and not fast_iter:
        post_prune_n_list = []
        most_playouts_child = tree.get_node(max_id)
        most_playouts_CPUCT = most_playouts_child.data.value_avg + config.CPUCT * most_playouts_child.data.policy * math.sqrt(root.data.visit_count) / (config.DPUCT + most_playouts_child.data.visit_count)

        for root_child_id in root_children_id:
            if root_child_id != max_id:
                root_child = tree.get_node(root_child_id)
                if root_child.data.visit_count > 0:
                    root_child_n_forced = math.sqrt(config.CForcedPlayout * root_child.data.policy * root.data.visit_count)
                    count = 0
                    while True:
                        if root_child.data.visit_count == 1:
                            root_child.data.visit_count = 0
                            break
                        root_child_CPUCT_minus = root_child.data.value_avg + config.CPUCT * root_child.data.policy * math.sqrt(root.data.visit_count) / (config.DPUCT + root.data.visit_count)
                        if count < root_child_n_forced and root_child_CPUCT_minus < most_playouts_CPUCT:
                            count += 1
                            root_child.data.visit_count -= 1
                        else:
                            break
            post_prune_n_list.append(tree.get_node(root_child_id).data.visit_count)

    save_move = not fast_iter
    return move, tree, save_move


def pick_random_move_by_policy(tree: MCTSTree) -> tuple:
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

def get_move_list(move_matrix, policy_matrix):
    # Returns list of possible moves with their policy
    # Removes buffer
    move_list = np.argwhere(move_matrix)

    # Formats moves from (policy index, row, col) to (value, (policy index, col - 2, row))
    move_list = [(policy_matrix[pi, ri, ci], (pi, ci - 2, ri)) for pi, ri, ci in move_list]

    return move_list

##### Neural Network #####
# 
# Player orientation: Active player, other player
# y:
#   Policy: (19 x 25 x 11) = 5225 (Hold x (Rows - 1) x (Columns + 1) x Rotations)
#   Value: (1)

def instantiate_network(config: Config, show_summary=True, save_network=True, plot_model=False):
    # Creates a network with random weights
    # 1: For each grid, apply the same neural network, and then use 1x1 kernel and concatenate
    # 1 -> 2: For opponent grid, apply fully connected layer
    # Concatenate active player's kernels/features with opponent's dense layer and non-player specific features
    # Apply value head and policy head 

    if config.model == 'keras':
        if isinstance(config.model_config, AuxBaseResNetConfig):
            model = gen_auxbaseresnet_keras(config.model_config, config.use_tanh)
        elif isinstance(config.model_config, BaseResNetConfig):
            model = gen_baseresnet_keras(config.model_config, config.use_tanh)
        else:
            model = gen_alphasame_nn(config.model_config, config.use_tanh)
    elif config.model == 'pytorch':
        if isinstance(config.model_config, AuxBaseResNetConfig):
            model = AuxBaseResNet(config.model_config, config.use_tanh)
        elif isinstance(config.model_config, BaseResNetConfig):
            model = BaseResNet(config.model_config, config.use_tanh)
        else:
            model = AlphaSame(config.model_config, config.use_tanh)
        model.to(device)

    if config.model == 'keras':
        if plot_model == True:
            keras.utils.plot_model(model, to_file=f"{directory_path}/model_{config.model_version}_img.png", show_shapes=True)

        # Loss is the sum of MSE of values and Cross entropy of policies
        model.compile(optimizer=keras.optimizers.Adam(
            learning_rate=config.learning_rate),
            loss=["mean_squared_error", "categorical_crossentropy"],
            loss_weights=config.loss_weights
            )

        if show_summary: model.summary()

        if save_network:
            path = config.model_dir
            os.makedirs(path, exist_ok=True)
            model.save(f"{path}/0.keras")
            append_version_record(config, 0)

        return model
    elif config.model == 'pytorch':
        if show_summary: print(model)

        if save_network:
            path = config.model_dir
            os.makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), f"{path}/0.pt")
            append_version_record(config, 0)

        return model

def train_network(config, model, set):
    if config.model == 'keras':
        train_network_keras(config, model, set)
    elif config.model == 'pytorch':
        train_network_pytorch(config, model, set)

def train_network_keras(config, model, set, data_number=None):
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

    history = model.fit(x=features,
                        y=[values, policies],
                        batch_size=64,
                        epochs=config.epochs,
                        shuffle=config.shuffle)

    # Compute per-head losses manually (model.evaluate returns a single weighted scalar)
    preds = model.predict(features, verbose=0)
    value_preds = preds[0].flatten()
    policy_preds = preds[1]
    value_loss = float(np.mean((value_preds - values.flatten()) ** 2))
    policy_loss = float(-np.mean(np.sum(policies * np.log(np.clip(policy_preds, 1e-7, 1.0)), axis=-1)))

    log_entry = {
        "model_version": config.model_version,
        "data_number": data_number if data_number is not None else highest_data_number(config),
        "loss": float(history.history['loss'][-1]),
        "value_loss": value_loss,
        "policy_loss": policy_loss,
    }
    with open(f"{logs_path}/training_log.jsonl", 'a') as f:
        f.write(ujson.dumps(log_entry) + '\n')

def train_network_pytorch(config, model, set, data_number=None):
    features = list(map(list, zip(*set)))

    for i in range(len(features)):
        features[i] = torch.tensor(features[i])

        if features[i].dim() == 3 and features[i].shape[1] == ROWS and features[i].shape[2] == COLS:
            features[i] = features[i].unsqueeze(dim=1).type(torch.float)

    dataset = torch.utils.data.TensorDataset(*features)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)

    loss_fn_value = nn.MSELoss()
    loss_fn_policy = nn.CrossEntropyLoss()
    loss_fn_aux = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    has_aux = isinstance(config.model_config, AuxBaseResNetConfig)
    aux_weight = config.model_config.aux_weight if has_aux else 0.0

    model.train()
    total_loss = value_loss_sum = policy_loss_sum = aux_loss_sum = 0.0
    batches = 0

    for _ in range(config.epochs):
        for data_point in dataloader:
            data_point = [feat.to(device) for feat in data_point]

            y_policy = data_point.pop().reshape(-1, POLICY_SIZE)
            y_value = data_point.pop().type(torch.float)

            if has_aux:
                y_aux = compute_aux_targets(data_point[0])
                pred_value, pred_policy, pred_aux = model(*data_point)
            else:
                pred_value, pred_policy = model(*data_point)

            pred_value = pred_value.reshape(-1)
            pred_policy = pred_policy.reshape(-1, POLICY_SIZE)

            loss_1 = loss_fn_value(pred_value, y_value)
            loss_2 = loss_fn_policy(pred_policy, y_policy)
            loss = loss_1 + loss_2

            if has_aux:
                loss_3 = loss_fn_aux(pred_aux, y_aux)
                loss = loss + aux_weight * loss_3
                aux_loss_sum += loss_3.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            value_loss_sum += loss_1.item()
            policy_loss_sum += loss_2.item()
            total_loss += loss.item()
            batches += 1

    model.eval()

    avg_loss = total_loss / max(batches, 1)
    avg_value_loss = value_loss_sum / max(batches, 1)
    avg_policy_loss = policy_loss_sum / max(batches, 1)
    avg_aux_loss = aux_loss_sum / max(batches, 1)

    if has_aux:
        print(f"loss: {avg_loss:>7f}  value: {avg_value_loss:>7f}  policy: {avg_policy_loss:>7f}  aux: {avg_aux_loss:>7f}")
    else:
        print(f"loss: {avg_loss:>7f}  value: {avg_value_loss:>7f}  policy: {avg_policy_loss:>7f}")

    log_entry = {
        "model_version": config.model_version,
        "data_number": data_number if data_number is not None else highest_data_number(config),
        "loss": avg_loss,
        "value_loss": avg_value_loss,
        "policy_loss": avg_policy_loss,
        "backend": "pytorch",
    }
    if has_aux:
        log_entry["aux_loss"] = avg_aux_loss
    with open(f"{logs_path}/training_log.jsonl", 'a') as f:
        f.write(ujson.dumps(log_entry) + '\n')

def evaluate(config, game, network):
    if config.model == 'keras':
        if config.use_tflite:
            # Use tflite interpreter
            return evaluate_from_tflite(game, network)
        else:
            return evaluate_from_keras(game, network)
    elif config.model == 'pytorch':
        return evaluate_pytorch(game, network)
    
def evaluate_from_tflite(game, interpreter):
    # Use a neural network to return value and policy.

    # Build metadata cache once per interpreter instance (never changes after allocation).
    if not hasattr(interpreter, '_eval_cache'):
        input_details  = interpreter.get_input_details()
        output_details = sorted(interpreter.get_output_details(), key=lambda x: x['name'])
        idx_map = []
        for det in input_details:
            split_str = det['name'].split(":")[0]
            if len(split_str) == 12:   # "serving_default" → first input
                idx_map.append(0)
            else:
                idx_map.append(int(split_str.split("_")[2]))
        interpreter._eval_cache = {
            'idx_map': idx_map,
            'val_idx': output_details[0]['index'],
            'pol_idx': output_details[1]['index'],
        }
    cache = interpreter._eval_cache

    data = game_to_X(game)
    X = []
    for feature in data:
        if type(feature) in (float, int):
            X.append(np.expand_dims(np.float32(feature), axis=(0, 1)))
        else:
            # np.asarray avoids a copy when feature is already float32 ndarray
            np_feature = np.expand_dims(np.asarray(feature, dtype=np.float32), axis=0)
            if np_feature.shape == (1, ROWS, COLS): # Expand grids
                np_feature = np.expand_dims(np_feature, axis=-1)
            X.append(np_feature)

    for i, idx in enumerate(cache['idx_map']):
        interpreter.set_tensor(i, X[idx])

    interpreter.invoke()

    value    = interpreter.get_tensor(cache['val_idx']).item()
    policies = interpreter.get_tensor(cache['pol_idx']).reshape(POLICY_SHAPE)

    return value, policies

def evaluate_from_keras(game, model):
    # Use a neural network to return value and policy.
    data = game_to_X(game)
    X = []
    for feature in data:
        if type(feature) in (float, int):
            X.append(np.expand_dims(np.float32(feature), axis=(0, 1)))
        else:
            np_feature = np.expand_dims(np.float32(feature), axis=0)
            if np_feature.shape == (1, ROWS, COLS): # Expand grids
                np_feature = np.expand_dims(np_feature, axis=-1)
            X.append(np_feature)
        
    out = model.predict_on_batch(X)
    # AuxBaseResNet (3 outputs) returns [value, policy, aux]; ignore aux at inference.
    value, policies = out[0], out[1]
    value = value.item()
    policies = policies.reshape(POLICY_SHAPE)

    return value, policies

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

        out = model.forward(*X)
        value, policies = out[0], out[1]  # AuxBaseResNet returns a 3-tuple; ignore aux at inference

    policies = torch.softmax(policies.reshape(-1), dim=0)
    return value.item(), policies.cpu().numpy().reshape(POLICY_SHAPE)

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

            assert policy_index >= 0 and row >= 0 and col + 2 >= 0 # Make sure indices are nonnegative

            # ACCOUNT FOR BUFFER
            probability_matrix[policy_index][row][col + 2] = round(root_child_n / total_n, 4)

    return probability_matrix


def simplify_grid(grid):
    # Replaces minos in a grid with 1s, returning a float32 numpy array.
    return (grid != 0).astype(np.float32)

# Helper function to reverse data if needed
def reverse_if_needed(data, condition):
    return data[::-1] if condition else data

# Methods for getting game data
# All of them orient the info in the perspective of the active player
def get_grids(game):
    # simplify_grid now creates float32 numpy arrays directly; no copy needed.
    grids = [simplify_grid(player.board.grid) for player in game.players]
    return reverse_if_needed(grids, game.turn == 1)

def get_pieces(game):
    piece_table = np.zeros((2, 2 + PREVIEWS, len(MINOS)), dtype=np.float32)
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

def get_garbage(game):
    garbage = [len(player.garbage_to_receive) for player in game.players]
    return reverse_if_needed(garbage, game.turn == 1)

def game_to_X(game):
    # Returns game information for the network.
    # Orient all info in perspective to the current player
    grids = get_grids(game)
    pieces = get_pieces(game)
    b2b = get_stat(game, 'b2b')
    combo = get_stat(game, 'combo')
    garbage = get_garbage(game)
    color = game.players[game.turn].color

    return (
        grids[0], pieces[0], b2b[0], combo[0], garbage[0],
        grids[1], pieces[1], b2b[1], combo[1], garbage[1],
        color
    )

    '''
    return (
        grids[0], pieces[0], b2b[0], combo[0], lines_cleared[0], lines_sent[0], 
        grids[1], pieces[1], b2b[1], combo[1], lines_cleared[1], lines_sent[1], 
        color, pieces_placed
    )
    '''

def reflect_grid(grid):
    # Return a grid flipped horizontally
    if isinstance(grid, np.ndarray):
        return np.fliplr(grid).tolist()
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
        piece, rotation, t_spin_index = policy_index_to_piece[policy_index]

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

        new_policy_index = policy_piece_to_index[new_piece][new_rotation][t_spin_index]
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

def get_game_stats(config, game, data_number, model_number):
    # Returns a dictionary with player statistics
    stats = {
        "model_number": model_number,
        "model_version": config.model_version,
        "data_number": data_number,
        "data_version": config.data_version,
        "app": game.players[0].stats.lines_sent  / game.players[0].stats.pieces,
        "dspp": game.players[0].stats.lines_cleared / game.players[0].stats.pieces
    }

    return stats

def col_heights_from_grid(grid):
    rows, cols = len(grid), len(grid[0])
    col_heights = np.zeros(cols, dtype=np.int32)
    for c in range(cols):
        first = -1
        for r in range(rows):
            if grid[r][c] != 0:
                first = r
                break
        col_heights[c] = (rows - first) if first != -1 else 0
    return col_heights

def count_holes(grid, col_heights):
    rows, cols = len(grid), len(grid[0])
    holes = 0
    for c in range(cols):
        if col_heights[c] == 0:
            continue
        top = rows - col_heights[c]
        for r in range(top, rows):
            if grid[r][c] == 0:
                holes += 1
    return holes

def calculate_board_metrics(grid):
    col_heights = col_heights_from_grid(grid)
    avg_h = int(col_heights.sum())

    holes = count_holes(grid, col_heights)

    # simple normalizations
    holes_norm = min(1.0, holes / float((ROWS - 1) * COLS))
    avg_norm   = min(1.0, avg_h / float(ROWS * COLS))

    return {
        "holes": holes_norm,
        "avg_height": avg_norm,
    }

def play_game(config, interference_network, game_number=None, screen=None):
    # AI plays one game against itself
    # Returns a tuple: (game_data, player_stats)
    if config.visual == True:
        if screen == None:
            screen = pygame.display.set_mode( (WIDTH, HEIGHT))
        pygame.display.set_caption(f'Training game {game_number}')

        pygame.event.get()

    game = Game(config.ruleset)
    game.setup()

    # Initialize data storage
    # Each player's move data will be stored in their respective list
    game_data = [[], []]

    # Initialize the game with random moves
    if config.use_random_starting_moves:
        # Because dirichlet alpha is scaled with dirichlet_s and not the
        # board size, use dirichlet_s as a general scaling factor proportional
        # to the action space
        scale = 0.04 * config.DIRICHLET_S
        num_random_moves = np.random.exponential(scale=scale)

        fast_config = config.copy()
        fast_config.MAX_ITER = 1
        fast_config.use_playout_cap_randomization = False
        fast_config.use_dirichlet_noise = False
        fast_config.use_forced_playouts_and_policy_target_pruning = False

        while num_random_moves > 0 and game.is_terminal == False:
            # Play random moves proportional to the raw policy distribution
            _, temp_tree, _ = MCTS(fast_config, game, interference_network)

            move = pick_random_move_by_policy(temp_tree)
            game.make_move(move)

            num_random_moves -= 1

    while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
        move, tree, save = MCTS(config, game, interference_network)

        if save or config.save_all:
            search_matrix = search_statistics(tree) # Moves that the network looked at
            
            # Get data
            move_data = [*game_to_X(game)] 

            if config.augment_data: 
                reflected_search_matrix = reflect_policy(search_matrix)
                
                # Reflect each player for a total of 2 * 2 = 4 times more data
                # When the other player is reflected, it shouldn't impact active player
                # When the active player is reflected, the search placements need to be reflected as well
                for active_player_idx in range(2): # 0: not reflected, 1: reflected
                    for other_player_idx in range(2):
                        # Copy move data
                        copied_data = []
                        for feature in move_data:
                            # Copy data
                            if isinstance(feature, np.ndarray): # queue array
                                copied_data.append(feature.copy())
                            elif type(feature) == list: # board
                                copied_data.append([x[:] for x in feature])
                            else: # int or float
                                copied_data.append(feature)

                        # Flip boards and pieces
                        if active_player_idx == 1:
                            copied_data[0] = reflect_grid(copied_data[0])
                            copied_data[1] = reflect_pieces(copied_data[1])
                        
                        if other_player_idx == 1:
                            copied_data[5] = reflect_grid(copied_data[5])
                            copied_data[6] = reflect_pieces(copied_data[6])

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

        if config.visual == True:
            game.show(screen)
            pygame.display.update()

    # After game ends update value
    winner = game.winner
    for player_idx in range(len(game_data)):
        if winner == -1: # Draw
            value = config.value_mid # draw 0 or 0.5
        elif winner == player_idx:
            value = config.value_max # win 1 or 1
        else:
            value = config.value_min # loss -1 or 0
        # Insert value before policy for each move of that player
        for move_idx in range(len(game_data[player_idx])):
            game_data[player_idx][move_idx].insert(-1, value)

            # Add auxiliary targets
            if False:
                board_metrics = calculate_board_metrics(game_data[player_idx][move_idx][0])
                game_data[player_idx][move_idx].append(board_metrics["holes"])
                game_data[player_idx][move_idx].append(board_metrics["avg_height"])

    # Reformat data to stack all moves into one continuous list
    data = game_data[0]
    data.extend(game_data[1])

    stats = get_game_stats(config, game, highest_data_number(config) + 1, highest_model_number(config))

    return data, stats


async def aplay_game(config, evaluator, game_number=None, screen=None):
    """Async mirror of play_game. Uses amcts + BatchedEvaluator. Pygame rendering
    only when `screen` is non-None (caller decides which single game gets the window)."""
    do_render = (screen is not None and config.visual)
    if do_render:
        pygame.display.set_caption(f'Training game {game_number}')
        pygame.event.get()

    game = Game(config.ruleset)
    game.setup()

    game_data = [[], []]

    if config.use_random_starting_moves:
        scale = 0.04 * config.DIRICHLET_S
        num_random_moves = np.random.exponential(scale=scale)

        fast_config = config.copy()
        fast_config.MAX_ITER = 1
        fast_config.use_playout_cap_randomization = False
        fast_config.use_dirichlet_noise = False
        fast_config.use_forced_playouts_and_policy_target_pruning = False

        while num_random_moves > 0 and game.is_terminal == False:
            _, temp_tree, _ = await amcts(fast_config, game, evaluator)
            move = pick_random_move_by_policy(temp_tree)
            game.make_move(move)
            num_random_moves -= 1

    while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
        move, tree, save = await amcts(config, game, evaluator)

        if save or config.save_all:
            search_matrix = search_statistics(tree)
            move_data = [*game_to_X(game)]

            if config.augment_data:
                reflected_search_matrix = reflect_policy(search_matrix)

                for active_player_idx in range(2):
                    for other_player_idx in range(2):
                        copied_data = []
                        for feature in move_data:
                            if isinstance(feature, np.ndarray):
                                copied_data.append(feature.copy())
                            elif type(feature) == list:
                                copied_data.append([x[:] for x in feature])
                            else:
                                copied_data.append(feature)

                        if active_player_idx == 1:
                            copied_data[0] = reflect_grid(copied_data[0])
                            copied_data[1] = reflect_pieces(copied_data[1])

                        if other_player_idx == 1:
                            copied_data[5] = reflect_grid(copied_data[5])
                            copied_data[6] = reflect_pieces(copied_data[6])

                        for i in range(len(copied_data)):
                            if isinstance(copied_data[i], np.ndarray):
                                copied_data[i] = copied_data[i].tolist()

                        if active_player_idx == 0:
                            copied_data.append(search_matrix)
                        else:
                            copied_data.append(reflected_search_matrix)

                        game_data[game.turn].append(copied_data)
            else:
                for i in range(len(move_data)):
                    if isinstance(move_data[i], np.ndarray):
                        move_data[i] = move_data[i].tolist()

                move_data.append(search_matrix)
                game_data[game.turn].append(move_data)

        game.make_move(move)

        if do_render:
            game.show(screen)
            pygame.display.update()
            await asyncio.sleep(0)  # yield to other coroutines

    winner = game.winner
    for player_idx in range(len(game_data)):
        if winner == -1:
            value = config.value_mid
        elif winner == player_idx:
            value = config.value_max
        else:
            value = config.value_min
        for move_idx in range(len(game_data[player_idx])):
            game_data[player_idx][move_idx].insert(-1, value)

            if False:
                board_metrics = calculate_board_metrics(game_data[player_idx][move_idx][0])
                game_data[player_idx][move_idx].append(board_metrics["holes"])
                game_data[player_idx][move_idx].append(board_metrics["avg_height"])

    data = game_data[0]
    data.extend(game_data[1])

    stats = get_game_stats(config, game, highest_data_number(config) + 1, highest_model_number(config))

    return data, stats


def make_training_set(config, interference_network, num_games, save_game=False, save_stats=True, screen=None):
    # Creates a dataset of several AI games.
    if config.model == 'pytorch' and config.batched_inference:
        series_data, series_stats = asyncio.run(_make_training_set_async(
            config, interference_network, num_games, screen))
    else:
        series_data = []
        series_stats = []
        for idx in range(1, num_games + 1):
            data, stats = play_game(config, interference_network, game_number=idx, screen=screen)
            series_data.extend(data)
            series_stats.append(stats)

    if save_game == True:
        json_data = ujson.dumps(series_data)

        # Increment set counter
        next_set = highest_data_number(config) + 1

        with open(f"{config.data_dir}/{next_set}.txt", 'w') as out_file:
            out_file.write(json_data)

    if save_stats == True:
        averaged_stats = {
            "model_number": series_stats[0]["model_number"],
            "model_version": series_stats[0]["model_version"],
            "data_number": series_stats[0]["data_number"],
            "data_version": series_stats[0]["data_version"],
            "app": round(sum([x["app"] for x in series_stats]) / len(series_stats), 3),
            "dspp": round(sum([x["dspp"] for x in series_stats]) / len(series_stats), 3)
        }

        with open(f"{logs_path}/stats.jsonl", 'a') as out_file:
            out_file.write(ujson.dumps(averaged_stats) + '\n')

    else:
        return series_data


async def _make_training_set_async(config, model, num_games, screen):
    """Run num_games concurrently with shared BatchedEvaluator. Only game 1
    receives a pygame screen (others run headless)."""
    evaluator = BatchedEvaluator(model, config, max_batch=num_games)
    await evaluator.start()
    try:
        coros = [
            aplay_game(config, evaluator,
                       game_number=idx,
                       screen=(screen if idx == 1 else None))
            for idx in range(1, num_games + 1)
        ]
        results = await asyncio.gather(*coros)
    finally:
        await evaluator.stop()

    series_data = []
    series_stats = []
    for data, stats in results:
        series_data.extend(data)
        series_stats.append(stats)
    return series_data, series_stats

def load_data_and_train_model(config, model, data=None):
    path = config.data_dir

    if config.data_loading_style == "merge":
        if data == None:
            data = []
            filenames = get_data_filenames(config)

            filenames.sort(key=lambda x: int(x.split('.')[0]), reverse=True) # Sort by descending order

            age = 0

            for filename in filenames:
                weight_decay = config.decay_factor ** age

                set = ujson.load(open(f"{path}/{filename}", 'r'))

                data.extend(random.sample(set, int(len(set) * weight_decay))) # Randomly sample

                age += 1
        else:
            print("Not using weight decay")
            data = [x for set in data for x in set] # Flatten list

        if config.shuffle == True:
            random.shuffle(data)

        print(f"Training with {len(data)} samples over {len(filenames)} sets with decay factor {config.decay_factor}")
        train_network(config, model, data)

        gc.collect()

    elif config.data_loading_style == 'distinct':
        raise Exception("Not fully fleshed out")
        if data == None:
            data = load_data(config)

            for set in data:
                print(f"Training with {len(set)} samples")
                train_network(config, model, set)

        else:
            if config.shuffle == True:
                random.shuffle(data)
            for set in data:
                print(f"Training with {len(set)} samples")
                train_network(config, model, set)

        gc.collect()
    
    else: raise NotImplementedError

def load_data(config) -> list:
    # Load data from the past n games
    # Returns a list where each indice is a set

    data = []

    sets, len_sets = 0, 0

    path = config.data_dir

    # Get filenames and load them
    for filename in get_data_filenames(config):
        set = ujson.load(open(f"{path}/{filename}", 'r'))
        data.append(set)

        sets += 1
        len_sets += len(set)
    
    print(sets, len_sets)
    return data

def get_data_filenames(config) -> list:
    # Returns a list of data filenames
    filenames = []
    max_set = highest_data_number(config)

    sets = 0

    path = config.data_dir

    # Get n games
    for filename in os.listdir(path):
        try:
            # Ignore files that don't start with numbers
            if filename.split('.')[0].isdigit() == False:
                continue

            data_number = int(filename.split('.')[0])
            if data_number > max_set - config.sets_to_train_with:
                # Load data
                filenames.append(filename)

                sets += 1
        except (ValueError, IndexError):
            continue
    
    # print(sets)
    # Shuffle data
    if config.shuffle:
        random.shuffle(filenames)
    return filenames

def battle_networks(NN_1, config_1, NN_2, config_2, threshold, threshold_type, games, network_1_title='Network 1', network_2_title='Network 2', screen=None):
    # Battle two AI networks and returns results with optional early termination
    # Returns tuple of (wins_array, True if NN_1 met the threshold, False otherwise)
    if config_1.ruleset != config_2.ruleset:
        raise NotImplementedError("Ruleset's aren't equal")

    use_async = (
        config_1.model == 'pytorch' and config_2.model == 'pytorch'
        and config_1.batched_inference and config_2.batched_inference
    )
    if use_async:
        wins = asyncio.run(_battle_networks_async(NN_1, config_1, NN_2, config_2, games))
        accepted = _check_threshold(wins, games, threshold, threshold_type)
        return wins, accepted

    wins = np.zeros((2), dtype=int)
    flip_color = False
    visual = config_1.visual and config_2.visual

    for i in range(games):
        if visual:
            if screen == None:
                screen = pygame.display.set_mode( (WIDTH, HEIGHT))

            if not flip_color:
                title = f'{network_1_title} | {wins[0]} vs {wins[1]} | {network_2_title}'
            else:
                title = f'{network_2_title} | {wins[1]} vs {wins[0]} | {network_1_title}'
            pygame.display.set_caption(title)

            pygame.event.get()

        game = Game(config_1.ruleset)
        game.setup()

        while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
            if (game.turn == 0 and not flip_color) or (game.turn == 1 and flip_color):
                move, *_ = MCTS(config_1, game, NN_1)
            else:
                move, *_ = MCTS(config_2, game, NN_2)
            game.make_move(move)

            if visual == True:
                game.show(screen)
                pygame.display.update()

        winner = game.winner
        if winner == -1:
            wins += 0.5
        elif not flip_color: # ARGGHGHHHH
            wins[winner] += 1
        else: # If the color is flipped, nn_1 is playing for player 2, and nn_2 is playing for player 1
            wins[1 - winner] += 1

        flip_color = not flip_color

        # Terminate early if threshold is provided and either network meets it
        if threshold is not None:
            if threshold_type == 'more':
                if wins[0] > threshold * games:
                    # print(*wins)
                    return wins, True
                elif wins[1] >= (1 - threshold) * games:
                    # print(*wins)
                    return wins, False
            elif threshold_type == 'moreorequal':
                if wins[0] >= threshold * games:
                    # print(*wins)
                    return wins, True
                elif wins[1] > (1 - threshold) * games:
                    # print(*wins)
                    return wins, False
        # else:
            # raise NotImplementedError

    # If neither side eaches a cutoff, return None
    # print(*wins)
    return wins, None


def _check_threshold(wins, games, threshold, threshold_type):
    """Post-hoc threshold check for async battle (no early termination)."""
    if threshold is None:
        return None
    if threshold_type == 'more':
        if wins[0] > threshold * games:
            return True
        if wins[1] >= (1 - threshold) * games:
            return False
    elif threshold_type == 'moreorequal':
        if wins[0] >= threshold * games:
            return True
        if wins[1] > (1 - threshold) * games:
            return False
    return None


async def aplay_battle_game(config_1, config_2, ev1, ev2, side):
    """Play one game between two networks. `side=0` → ev1 plays player 0;
    `side=1` → ev1 plays player 1. Returns (winner_idx, side)."""
    game = Game(config_1.ruleset)
    game.setup()

    while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
        if (game.turn == 0 and side == 0) or (game.turn == 1 and side == 1):
            move, *_ = await amcts(config_1, game, ev1)
        else:
            move, *_ = await amcts(config_2, game, ev2)
        game.make_move(move)

    return game.winner, side


async def _battle_networks_async(NN_1, config_1, NN_2, config_2, games):
    """Run `games` battles concurrently with two BatchedEvaluators (one per network).
    No early termination — all games play out. Returns wins array."""
    ev1 = BatchedEvaluator(NN_1, config_1, max_batch=games)
    ev2 = BatchedEvaluator(NN_2, config_2, max_batch=games)
    await ev1.start()
    await ev2.start()
    try:
        coros = [
            aplay_battle_game(config_1, config_2, ev1, ev2, side=i % 2)
            for i in range(games)
        ]
        results = await asyncio.gather(*coros)
    finally:
        await ev1.stop()
        await ev2.stop()

    wins = np.zeros((2), dtype=float)
    for winner, side in results:
        if winner == -1:
            wins[0] += 0.5
            wins[1] += 0.5
        elif side == 0:
            wins[winner] += 1
        else:
            wins[1 - winner] += 1
    return wins


def self_play_loop(config, skip_first_set=False):
    screen = None
    if config.visual == True:
        screen = pygame.display.set_mode( (WIDTH, HEIGHT))
    # Given a network, generates training data, trains it, and checks if it improved.
    best_train_network, best_interference_network = load_best_train_and_interference_models(config)

    training_config = config.copy()
    # Setting training to true enables a variety of training features
    training_config.training = True

    iter = 0

    while True:
        iter += 1

        # The challenger network will be trained and then battled against the prior network
        challenger_train_network, challenger_interference_network = load_best_train_and_interference_models(config)

        # Play a training set and train the network on past sets.
        if not skip_first_set:
            print(f"Starting training loop {iter} with network version {highest_model_number(config)}")
            for i in range(config.training_loops):
                # Make data file with trianing config
                make_training_set(training_config, challenger_interference_network, num_games=config.training_games, save_game=True, save_stats=True, screen=screen)
                print("Finished making training set")
        else:
            # Skip and stop skipping in future
            skip_first_set = False

        # Load data and train
        load_data_and_train_model(config, challenger_train_network, data=None)
        challenger_interference_network = get_interference_network(config, challenger_train_network)

        print("Finished training network")

        # If new network is improved, save it and make it the default
        # Otherwise, repeat
        next_ver = highest_model_number(config) + 1
        print(f"Battling a challenger against version {next_ver - 1}")

        win_loss, win = battle_networks(
            challenger_interference_network, 
            config, 
            best_interference_network, 
            config, 
            config.gating_threshold, 
            config.gating_threshold_type,
            config.battle_games, 
            network_1_title='Challenger',
            network_2_title='Best',
            screen=screen
        )

        print(f"Challenger {win_loss[0]} - {win_loss[1]} Best")

        gating_entry = {
            "model_version": config.model_version,
            "challenger_number": next_ver,
            "challenger_wins": int(win_loss[0]),
            "best_wins": int(win_loss[1]),
            "total_games": config.battle_games,
            "win_rate": float(win_loss[0]) / config.battle_games,
            "accepted": bool(win),
        }
        with open(f"{logs_path}/gating_log.jsonl", 'a') as f:
            f.write(ujson.dumps(gating_entry) + '\n')

        if win:
            # Challenger network becomes next highest version
            print(f"Challenger version {next_ver} won and is now the best network")

            if config.model == 'keras':
                challenger_train_network.save(f"{config.model_dir}/{next_ver}.keras")
            if config.model == 'pytorch':
                torch.save(challenger_train_network.state_dict(), f"{config.model_dir}/{next_ver}.pt")
            append_version_record(config, next_ver)

            # The new network becomes the network to beat
            best_interference_network = challenger_interference_network
        
        gc.collect()

def get_interference_network(config, training_network):
    """Returns an interference network for the given training network."""

    if config.use_tflite and config.model == 'keras':
        # Load the model as a tflite interpreter
        return get_interpreter(training_network, num_threads=config.tflite_num_threads)

    else:
        # Load the model as a keras or pytorch model
        return training_network

def load_train_and_interference_models(config, model_number):
    """Loads the model and returns it along with an interpreter if needed."""

    if config.use_tflite and config.model == 'keras':
        # Load the model as a tflite interpreter
        model = load_model(config, model_number)
        interpreter = get_interpreter(model, num_threads=config.tflite_num_threads)
        return model, interpreter

    else:
        # Load the model as a keras or pytorch model
        model = load_model(config, model_number)
        return model, model

def load_best_train_and_interference_models(config):
    # Returns the model with the highest number
    max_ver = highest_model_number(config)

    return load_train_and_interference_models(config, max_ver)

def load_model(config, model_number):
    # Returns the model with the given number
    blockPrint()

    if config.model == 'keras':
        path = f"{config.model_dir}/{model_number}.keras"

        model = keras.models.load_model(path)
    elif config.model == 'pytorch':
        path = f"{config.model_dir}/{model_number}.pt"
        if not os.path.exists(path):
            # Legacy checkpoints saved without an extension
            path = f"{config.model_dir}/{model_number}"

        if isinstance(config.model_config, AuxBaseResNetConfig):
            model = AuxBaseResNet(config.model_config, config.use_tanh)
        elif isinstance(config.model_config, BaseResNetConfig):
            model = BaseResNet(config.model_config, config.use_tanh)
        else:
            model = AlphaSame(config.model_config, config.use_tanh)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.to(device)
        model.eval()

    enablePrint()

    print(path)

    return model

def load_best_model(config):
    # Returns the model with the highest number
    max_ver = highest_model_number(config)

    return load_model(config, max_ver)

def get_interpreter(model, num_threads=2):
    # Save a model as saved model, then load it as tflite.
    # num_threads caps the CPU pool used by the interpreter. Default 2 keeps the
    # laptop responsive during self-play; the Step 2 benchmark showed
    # Optimize.DEFAULT (prod_quant) is essentially batch-invariant so giving
    # more threads has marginal upside but real responsiveness cost.
    blockPrint()

    path = f"{directory_path}/TEMP_MODELS/savedmodel"
    model.export(path)

    converter = tf.lite.TFLiteConverter.from_saved_model(path)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]

    converter.optimizations = [tf.lite.Optimize.DEFAULT] # Quantization
    # converter.target_spec.supported_types = [tf.float16] # Float quantization
    '''
    def representative_dataset():
        for data in load_data(last_n_sets=1)[0][:100]:
            yield data

    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.representative_dataset = representative_dataset
    '''
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model, num_threads=num_threads)
    interpreter.allocate_tensors()

    enablePrint()

    return interpreter

def highest_model_number(config):
    max = -1

    if config.model == 'keras':
        path = config.model_dir

        os.makedirs(path, exist_ok=True)

        for filename in os.listdir(path):
            try:
                model_number = int(filename.split('.')[0])
                if model_number > max:
                    max = model_number
            except ValueError:
                continue

    elif config.model == 'pytorch':
        path = config.model_dir

        os.makedirs(path, exist_ok=True)

        for filename in os.listdir(path):
            stem = filename[:-3] if filename.endswith('.pt') else filename
            try:
                model_number = int(stem)
                if model_number > max:
                    max = model_number
            except ValueError:
                continue

    return max

def highest_data_number(config):
    max = -1

    path = config.data_dir

    os.makedirs(path, exist_ok=True)

    for filename in os.listdir(path):
        try:
            if not filename.split('.')[0].isdigit():
                continue
            data_number = int(filename.split('.')[0])
            if data_number > max:
                max = data_number
        except (ValueError, IndexError):
            continue

    return max

# Debug Functions
# Disable
def blockPrint():
    if HIDE_PRINTS:
        sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    sys.__stderr__