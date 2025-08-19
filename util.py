from ai import *
from const import *
from game import Game
from piece import Piece
from player import Player

import ast
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
import time

util_t_spin_board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 'I'], [0, 0, 0, 0, 0, 0, 0, 0, 0, 'I'], [0, 0, 0, 0, 0, 0, 0, 'Z', 0, 'I'], [0, 0, 0, 'Z', 0, 0, 'Z', 'Z', 'S', 'I'], [0, 0, 'Z', 'Z', 0, 0, 'Z', 'T', 'S', 'S'], [0, 0, 'Z', 'L', 0, 0, 0, 'T', 'T', 'S'], ['J', 'L', 'L', 'L', 'S', 'S', 0, 'T', 'O', 'O'], ['J', 'J', 'J', 'S', 'S', 0, 0, 'J', 'O', 'O'], ['I', 'I', 'I', 'I', 0, 0, 0, 'J', 'J', 'J']]
util_z_spin_board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 'J'], [0, 0, 0, 0, 0, 0, 0, 0, 0, 'J'], ['L', 0, 'T', 0, 0, 0, 0, 0, 'J', 'J'], ['L', 'T', 'T', 0, 0, 0, 0, 0, 0, 'T'], ['L', 'L', 'T', 'O', 'O', 0, 0, 0, 'T', 'T'], ['O', 'O', 'S', 'O', 'O', 0, 0, 0, 'S', 'T'], ['O', 'O', 'S', 'S', 'O', 'O', 0, 0, 'S', 'S'], ['L', 'L', 'I', 'S', 'O', 'O', 'T', 0, 'S', 'S'], [0, 'J', 'I', 'I', 'O', 'O', 0, 0, 'S', 'S'], [0, 'J', 'I', 'I', 'O', 'O', 0, 'S', 'S', 0], ['J', 'J', 'I', 'I', 0, 'I', 'I', 0, 'T', 0], [0, 'S', 'S', 'I', 0, 'I', 'I', 'T', 'T', 0], ['S', 'S', 'Z', 'Z', 0, 'I', 'I', 0, 'T', 0], [0, 'T', 0, 'Z', 'Z', 'I', 'I', 0, 'L', 'L'], ['T', 'T', 0, 0, 'J', 'J', 'J', 0, 0, 'L'], [0, 'T', 0, 0, 0, 0, 'J', 0, 0, 'L'], [1, 1, 1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]]
util_move_algo_board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 'I', 0, 0, 0, 0, 0], [0, 0, 0, 0, 'I', 0, 0, 0, 0, 0], [0, 0, 0, 0, 'I', 0, 0, 0, 0, 0], [0, 0, 0, 0, 'I', 0, 0, 0, 0, 0]]
util_move_algo_board_2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ['Z', 'Z', 0, 0, 0, 0, 0, 0, 0, 0], [0, 'Z', 'Z', 0, 0, 0, 0, 'S', 'S', 0], ['O', 'O', 0, 0, 0, 0, 'S', 'S', 'O', 'O'], ['O', 'O', 'Z', 0, 'I', 'I', 'I', 'I', 'O', 'O'], [0, 'Z', 'Z', 'S', 'S', 0, 'I', 'I', 'I', 'I'], [0, 'Z', 'S', 'S', 'T', 'T', 'T', 0, 0, 0], [0, 0, 0, 0, 0, 'T', 0, 0, 0, 0], [0, 0, 0, 0, 0, 'J', 'J', 'J', 0, 0], [0, 0, 0, 0, 0, 0, 0, 'J', 0, 0], [0, 0, 0, 0, 0, 0, 0, 'L', 'L', 0], [0, 0, 0, 0, 0, 0, 0, 0, 'L', 0], [0, 0, 0, 0, 0, 0, 0, 0, 'L', 0]]

# ------------------------- Internal Functions -------------------------
def battle_royale(interpreters, configs, names, num_games) -> dict:
    screen = None
    visual = all([config.visual for config in configs])

    if visual == True:
        screen = pygame.display.set_mode( (WIDTH, HEIGHT))

    scores={name: {} for name in names}

    for i in range(len(interpreters)):
        for j in range(i + 1, len(interpreters)):
            if i != j:
                (score_1, score_2), _ = battle_networks(interpreters[i], configs[i], 
                                                   interpreters[j], configs[j],
                                                   None, None, # Set threshold to None
                                                   num_games, 
                                                   network_1_title=names[i], 
                                                   network_2_title=names[j], 
                                                   screen=screen)
                
                scores[names[i]][names[j]] = f"{score_1}-{score_2}"
                scores[names[j]][names[i]] = f"{score_2}-{score_1}"

    return scores

def make_piece_coord_starting_row_dict():
    player = Player()
    res = {}
    for piece_type in mino_coords_dict:
        top_row = []
        player.create_piece(piece_type)
        check_rotations = True
        if piece_type == "O":
            check_rotations = False

        phase_2_queue = deque()

        piece = player.piece

        # Phase 1
        location = (piece.x_0, piece.y_0, piece.rotation)
        piece.coordinates = piece.get_self_coords

        phase_2_queue.append((piece.x_0, piece.y_0, piece.rotation))

        if check_rotations:
            for i in range(1, 4):
                player.try_wallkick(i)

                x = piece.x_0
                y = piece.y_0
                o = piece.rotation

                phase_2_queue.append((x, y, o))

                if i != 3:
                    piece.x_0, piece.y_0, piece.rotation = location

        # Phase 2
        while len(phase_2_queue) > 0:
            location = phase_2_queue.popleft()
            top_row.append(location)

            for x_dir in [-1, 1]:
                piece.x_0, piece.y_0, piece.rotation = location
                piece.coordinates = piece.get_self_coords

                while player.can_move(piece, x_offset=x_dir):
                    x = piece.x_0 + x_dir
                    y = piece.y_0
                    o = piece.rotation

                    piece.x_0 = x
                    piece.coordinates = piece.get_self_coords

                    top_row.append((x, y, o))
        
        res[piece_type] = top_row
    
    return res

def get_attribute_list_from_tree(tree, attr):
    res = []

    root = tree.get_node("root")
    child_ids = root.successors(tree.identifier)

    for child_id in child_ids:
        child = tree.get_node(child_id)
        res.append(getattr(child.data, attr))

    return res

# ------------------------- Test Functions -------------------------

def plot_dirichlet_noise() -> None:
    # Finding different values of dirichlet alpha affect piece decisions
    alpha_values = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    # alpha_values = [0.4, 0.3, 0.2, 0.1]
    alpha_values = {alpha: {'n_same': 0, 'n_total': 0} for alpha in alpha_values}

    use_dirichlet_s=False

    c = Config()

    model = load_best_model(c)
    interpreter = get_interpreter(model)

    for _ in range(10):
        game = Game(c.ruleset)
        game.setup()

        while game.is_terminal == False:
            default_move, _, _ = MCTS(c, game, interpreter)

            for alpha_value in alpha_values:
                config = Config(DIRICHLET_ALPHA=alpha_value, use_dirichlet_s=use_dirichlet_s, training=True, use_playout_cap_randomization=False)
                move, _, _ = MCTS(config, game, interpreter)

                if move == default_move:
                    alpha_values[alpha_value]['n_same'] += 1
                
                alpha_values[alpha_value]['n_total'] += 1

            # To change the board, make the default move
            game.make_move(default_move)

    percent_dict = {alpha: 100*alpha_values[alpha]['n_same']/alpha_values[alpha]['n_total'] for alpha in alpha_values}    

    fig, ax = plt.subplots()

    ax.bar(range(len(percent_dict)), list(percent_dict.values()), align='center')
    ax.set_xticks(range(len(percent_dict)), list(percent_dict.keys()))

    if use_dirichlet_s:
        ax.set_xlabel(f"Dirichlet Alpha Values; Dirichlet S {c.DIRICHLET_S}")
        ax.set_ylabel("% of moves that were the same as without noise")

        plt.savefig(f"{directory_path}/tst_alpha_vals_s")
    else:
        ax.set_xlabel("Dirichlet Alpha Values")
        ax.set_ylabel("% of moves that were the same as without noise")

        plt.savefig(f"{directory_path}/tst_alpha_vals")

    return percent_dict

def view_visit_count_and_policy_with_and_without_dirichlet_noise() -> None:
    # Creates a graph of the policy distribution before and after dirichlet noise is applied
    c = Config(training=True,
               use_playout_cap_randomization=False, 
               use_forced_playouts_and_policy_target_pruning=False)

    g = Game(c.ruleset)
    g.setup()

    no_noise_config = c.copy()
    noisy_config = c.copy()
    noisy_config.use_dirichlet_noise = True

    interpreter = get_interpreter(load_best_model(c))
    _, no_noise_tree, _ = MCTS(no_noise_config, g, interpreter)
    _, noisy_tree, _ = MCTS(noisy_config, g, interpreter)

    root_child_n = get_attribute_list_from_tree(noisy_tree, "visit_count")
    pre_noise_policy = get_attribute_list_from_tree(no_noise_tree, "policy")
    post_noise_policy = get_attribute_list_from_tree(noisy_tree, "policy")

    fig, axs = plt.subplots(3)
    fig.suptitle('Policy and visit count before and after dirichlet noise')
    axs[0].plot(root_child_n)
    axs[1].plot(pre_noise_policy)
    axs[2].plot(post_noise_policy)
    plt.savefig(f"{directory_path}/visit_count_vs_policy_vs_policy+noise_{c.ruleset}_{c.model_version}.png")
    print("Saved")

def time_move_matrix(algo) -> None:
    # Test the game speed
    # Returns the average speed of each move over n games

    # Old test results:
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

    # ---------- 100 iter ----------
    # Initial:                        0.340 0.357 0.362
    # Deque:                          0.382
    # Deque + set:                    0.310
    # Pop first:                      0.320
    # Don't use array                 0.297
    #   Using mp Queue                0.354
    
    # Default                         0.298
    # Don't check o rotations         0.280
    # Use single softdrop             0.339
    # Using fast algo                 0.194

    # Without quantization            0.338
    # Random evaluation (no NN)       0.150
    #   Dynamic range quantization    0.251
    #     With harddrop algo          0.153
    #   Float16 quantization          0.352
    #   Experimental int16/int8       0.239

    # Using lookup table for row      0.239

    # Added s2 rules; 
    #   brute force                   0.729
    #   faster but loss               0.276

    #   conv-brute-force              0.268

    c = Config(MAX_ITER=100, move_algorithm=algo)

    num_games = 8

    # Initialize pygame
    screen = pygame.display.set_mode( (WIDTH, HEIGHT))
    pygame.display.set_caption(f'Profiling Get Move Matrix')

    interpreter = get_interpreter(load_best_model(c))

    moves = 0
    START = time.time()

    for _ in range(num_games):
        game = Game(c.ruleset)
        game.setup()

        while game.is_terminal == False:
            move, _, _ = MCTS(c, game, interpreter)
            game.make_move(move)
            moves += 1

            game.show(screen)
            pygame.event.get()
            pygame.display.update()

    END = time.time()

    print(f"Time: {(END-START)/moves}")

def time_architectures(var, values) -> None:
    # Tests how fast an algorithm takes to make a move
    scores = {}
    configs = [Config(MAX_ITER=100) for _ in range(len(values))]
    for value, config in zip(values, configs):
        setattr(config, var, value)
        network = get_interpreter(instantiate_network(config,show_summary=False, save_network=False))

        game = Game(configs[0].ruleset)
        game.setup()

        START = time.time()
        MCTS(config, game, network)
        END = time.time()

        scores[str(value)] = END - START
    print(scores)

def profile_game() -> None:
    c = Config()
    game = Game(c.ruleset)
    game.setup()

    network = get_interpreter(load_best_model(c))

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Testing algorithm accuracy')

    # Profile game
    with cProfile.Profile() as pr:
        play_game(c, network, 777, show_game=True, screen=screen)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(20)

def test_algorithm_accuracy(truth_algo='brute-force', test_algo='faster-but-loss') -> None:
    # Test how accurate an algorithm is
    # Returns the percent of moves an algorithm found compared to all possible moves

    num_games = 5

    # Initialize pygame
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Testing algorithm accuracy')

    c = Config()

    interpreter = get_interpreter(load_best_model(c))

    truth_moves = 0
    test_moves = 0

    for _ in range(num_games):
        game = Game(c.ruleset)
        game.setup()

        while game.is_terminal == False:
            truth_matrix = get_move_matrix(game.players[game.turn], algo=truth_algo)
            test_matrix = get_move_matrix(game.players[game.turn], algo=test_algo)

            truth_moves += np.sum(truth_matrix)
            test_moves += np.sum(test_matrix)

            # if test_moves != truth_moves:
            #     # diff = np.logical_xor(test_matrix, truth_matrix)
            #     # print(game.players[game.turn].board.grid)
            #     # print(np.argwhere(diff))

            # Make a move using the default algorithm
            move, _, _ = MCTS(c, game, interpreter)
            game.make_move(move)

            game.show(screen)
            pygame.event.get()
            pygame.display.update()
    
    print(f"Accuracy: {test_moves / truth_moves * 100}")

def visualize_piece_placements(game, moves, sleep_time=0.3):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Tetris')
    # Need to iterate through pygame events to initialize screen
    for event in pygame.event.get():
        pass

    # print(moves)
    for policy, move in moves:
        game_copy = game.copy()
        game_copy.make_move(move)

        game_copy.players[game_copy.turn].stats.attack_text += str(move)

        game_copy.show(screen)
        pygame.display.update()

        time.sleep(sleep_time)

def test_reflected_policy():
    # Testing if reflecting pieces, grids, and policy are accurate
    c = Config()

    game = Game(c.ruleset)
    game.setup()

    # Place a piece to make it more interesting
    # for i in range(10):
    #     game.place()

    game.players[game.turn].board.grid = util_t_spin_board
    
    game.players[game.turn].held_piece = "T"

    move_matrix = get_move_matrix(game.players[game.turn], algo='brute-force')
    moves = get_move_list(move_matrix, np.ones(shape=POLICY_SHAPE))

    visualize_piece_placements(game, moves, sleep_time=0.05)

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
                player.held_piece = MINOS[piece_row.tolist().index(1)]
        else:
            player.queue.pieces[idx - 2] = MINOS[piece_row.tolist().index(1)]
    
    # Reflect the policy and see if it matches
    reflected_move_matrix = reflect_policy(move_matrix)
    reflected_moves = get_move_list(reflected_move_matrix, np.ones(shape=POLICY_SHAPE))

    visualize_piece_placements(game, reflected_moves, sleep_time=0.5)

def visualize_get_move_matrix(c, board):
    game = Game(c.ruleset)
    game.setup()

    game.players[game.turn].piece = Piece(type="T")
    game.players[game.turn].piece.move_to_spawn()
    game.players[game.turn].held_piece = "T"
    game.players[game.turn].board.grid = board

    truth_matrix = get_move_matrix(game.players[game.turn], algo="brute-force")
    test_matrix = get_move_matrix(game.players[game.turn], algo="convolutional")

    truth_moves = np.sum(truth_matrix)
    test_moves = np.sum(test_matrix)

    print(truth_moves, test_moves)

    if test_moves != truth_moves:
        diff = np.logical_xor(test_matrix, truth_matrix)
        # print(game.players[game.turn].board.grid)
        print(np.argwhere(diff))

    move_matrix = get_move_matrix(game.players[game.turn], algo=c.move_algorithm)
    moves = get_move_list(move_matrix, np.ones(shape=POLICY_SHAPE))

    visualize_piece_placements(game, moves, sleep_time=1.0)

def test_parameters(
    var: str, 
    values: list,
    num_games: int,
    data=None,
    load_from_best_model: bool=False
):
    ## Grid search battling different parameters
    # Configs
    configs = [Config() for _ in range(len(values))]
    for value, config in zip(values, configs):
        setattr(config, var, value)

    test_configs(configs, num_games, data=data, load_from_best_model=load_from_best_model)

def test_configs(
    configs,
    num_games: int,
    data=None,
    load_from_best_model: bool=False
):
    ## Grid search battling different Configs

    # Networks
    if load_from_best_model:
        networks = [load_best_model(config) for config in configs]
    else:
        networks = [instantiate_network(config, show_summary=False, save_network=False, plot_model=False) for config in configs]

    if data != None:
        for config, network in zip(configs, networks):
             load_data_and_train_model(config, network, data)

    # Networks -> interpreters
    interpreters = [get_interference_network(config, network) for (config, network) in zip(configs, networks)]

    print(battle_royale(interpreters, 
                        configs, 
                        [f"Config {i+1}" for i in range(len(configs))], 
                        num_games))

def test_data_parameters(
    var: str, 
    values: list,
    learning_rate: float,
    num_training_loops: int,
    num_training_games: int,
    num_battle_games: int,
    load_from_best_model: bool = False,
):  
    # Configs
    # Set training to true
    configs = [Config(training=True, learning_rate=learning_rate) for _ in range(len(values))]
    for value, config in zip(values, configs):
        setattr(config, var, value)

    # Networks
    if load_from_best_model:
        networks = [load_best_model(config) for config in configs]
    else:
        networks = [instantiate_network(config, show_summary=False, save_network=False, plot_model=False) for config in configs]

    for config, network in zip(configs, networks):
        for _ in range(num_training_loops):
            interpreter = get_interpreter(network)
            set = make_training_set(config, interpreter, num_training_games, save_game=False)
            
            train_network(config, network, set)

            gc.collect()

    # Networks -> interpreters
    interpreters = [get_interpreter(network) for network in networks]

    # When battling, have each use the same config
    battle_configs = [Config() for _ in range(len(values))]

    print(battle_royale(interpreters, 
                        battle_configs, 
                        [str(value) for value in values], 
                        num_battle_games))
    
    print(var)

def test_network_versions(ver_1, ver_2):
    # Making sure that the newest iteration of a network is better than earlier versions
    c = Config()

    model_1 = get_interference_network(c, load_model(c, ver_1))
    model_2 = get_interference_network(c, load_model(c, ver_2))

    screen = pygame.display.set_mode( (WIDTH, HEIGHT))
    battle_networks(model_1, c, model_2, c, None, None, 200, f"Version {ver_1}", f"Version {ver_2}", screen)

def test_if_changes_improved_model():
    config = Config()
    network = instantiate_network(config, nn_generator=None, show_summary=False, save_network=False, plot_model=False)
    data = load_data(last_n_sets=10)

    for set in data:
        train_network_keras(config, network, set)

        gc.collect()

    best_nn = get_interpreter(load_best_model(config))
    chal_nn = get_interpreter(network)

    screen = pygame.display.set_mode( (WIDTH, HEIGHT))

    if battle_networks(chal_nn, config, best_nn, config, 0.55, 'moreorequal', 200, "New", "Best", show_game=True, screen=screen):
        print("Success")
    else:
        print("Failure")
    network.save(f"{directory_path}/New.keras")


def visualize_high_depth_replay(network, max_iter):
    # Battle an AI against itself at high depth, and then analyze it with undo and redo
    c=Config(MAX_ITER=max_iter)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Search Amount {max_iter} Replay Game")
    pygame.event.get() # Required for visuals?

    game = Game(c.ruleset)
    game.setup()

    while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
        move, _, _ = MCTS(c, game, network)
        game.make_move(move)

        game.show(screen)
        pygame.display.update()

    while True:
        game.show(screen)

        # Player's move:
        # Keyboard inputs
        for event in pygame.event.get():
            # Pressable at any time
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == k_undo:
                    game.undo()
                elif event.key == k_redo:
                    game.redo()
                elif event.key == k_restart:
                    game.restart()

        pygame.display.update()

def convert_data_and_train_4_7_to_4_8():
    config = Config(epochs=2)
    new_network = instantiate_network(config, show_summary=True, save_network=False)

    filenames = get_data_filenames(200, shuffle=False)

    path = f"{directory_path}/data/{4.7}"

    i = 0

    for filename in filenames:
        i += 1
        set = ujson.load(open(f"{path}/{config.ruleset}.{filename}", 'r'))
        # Manipulate the set
        '''
            grids[0], pieces[0], b2b[0], combo[0], garbage[0],
            grids[1], pieces[1], b2b[1], combo[1], garbage[1],
            color

            grids[0], pieces[0], b2b[0], combo[0], lines_cleared[0], lines_sent[0], 
            grids[1], pieces[1], b2b[1], combo[1], lines_cleared[1], lines_sent[1], 
            color, pieces_placed
        '''

        for move in set:
            move[4] = 0
            move[10] = 0
            move.pop(5)
            move.pop(10)
            move.pop(11)
    
        # Train challenger network
        train_network(config, new_network, set)
        gc.collect()
        
        if i % 10 == 0:
            new_network.save(f"{directory_path}/models/debug/{i}.keras")
        
    new_network.save(f"{directory_path}/models/debug/{i}.keras")
    
def convert_data_2_1_to_2_2(set):
    """
    Convert data from version 2.1 to 2.2.
    Resizes policy from (19, 25, 11) to (27, 25, 11).
    """
    # grids[0], pieces[0], b2b[0], combo[0], garbage[0],
    # grids[1], pieces[1], b2b[1], combo[1], garbage[1],
    # color, winner, search_matrix

    for move in set:
        policy = move[-1]
        t_policy = policy[-4:]
        copy_1 = copy.deepcopy(t_policy)
        copy_2 = copy.deepcopy(t_policy)
        policy.extend(copy_1)
        policy.extend(copy_2)

        # extra_policy = np.zeros(shape=(POLICY_SHAPE[0] - 19, POLICY_SHAPE[1], POLICY_SHAPE[2]), dtype=int).tolist()
        # move[-1].extend(extra_policy)

def convert_data_and_train(init_data_ver, conversion_function, last_n_sets, epochs):
    c = Config(epochs=epochs)

    new_network = instantiate_network(c, show_summary=True, save_network=False)

    filenames = get_data_filenames(c, data_ver=init_data_ver, last_n_sets=last_n_sets)

    path = f"{directory_path}/data/{c.ruleset}.{init_data_ver}"

    i = 0

    for filename in filenames:
        i += 1
        set = ujson.load(open(f"{path}/{filename}", 'r'))
        conversion_function(set)
    
        # Train challenger network
        train_network(c, new_network, set)
        gc.collect()
        
        if i % 5 == 0:
            new_network.save(f"{directory_path}/models/debug/{i}.keras")
        
    new_network.save(f"{directory_path}/models/debug/{i}.keras")

def visualize_policy():
    # Visualize the policy of a network
    c=Config(MAX_ITER=16)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Viewing policy of the network")
    pygame.event.get() # Required for visuals?

    game = Game(c.ruleset)
    game.setup()

    network = get_interpreter(load_best_model(c))

    # After a certain number of moves, the policy is examined
    moves = 10

    while game.is_terminal == False and len(game.history.states) < moves:
        move, _, _ = MCTS(c, game, network)
        game.make_move(move)

        game.show(screen)
        pygame.display.update()

    value, policy = evaluate(c, game, network)

    pygame.image.save(screen, f"{directory_path}/policy_visualization_screen.png")

    fig, axs = plt.subplots(1, POLICY_SHAPE[0], figsize=(40, 3))
    fig.suptitle('Policy visualization', y=0.98)
    for i in range(len(policy_index_to_piece)):
        axs[i].imshow(policy[i], cmap='viridis')
        if policy_index_to_piece[i][2] == 0:
            axs[i].set_title(f"{policy_index_to_piece[i][0]} rotation {policy_index_to_piece[i][1]}")
        else:
            axs[i].set_title(f"{policy_index_to_piece[i][0]} rotation {policy_index_to_piece[i][1]} {policy_index_to_piece[i][2]}")

    plt.savefig(f"{directory_path}/policy_visualization_{c.model_version}.png")
    print("saved")

def test_generate_move_matrix():
    c = Config()
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ['I', 0, 0, 0, 0, 0, 0, 0, 0, 'Z'], ['I', 0, 0, 0, 0, 0, 0, 0, 'Z', 'Z'], ['I', 0, 0, 0, 0, 0, 0, 0, 'Z', 'J'], ['I', 'T', 0, 'I', 'T', 0, 0, 0, 0, 'J'], ['T', 'T', 0, 'I', 'T', 'T', 'S', 0, 'J', 'J'], ['L', 'T', 'L', 'I', 'T', 0, 'S', 'S', 'S', 0], ['L', 0, 'L', 'I', 'J', 'O', 'O', 'S', 'S', 'S'], [0, 'T', 0, 'O', 'O', 'O', 'O', 0, 'Z', 'Z'], [0, 'T', 'T', 'O', 'O', 0, 'S', 'S', 'S', 'S'], [0, 'T', 'O', 'O', 0, 'S', 'S', 'S', 'S', 0], [0, 0, 'O', 'O', 0, 0, 0, 'Z', 'J', 0], [0, 0, 'I', 0, 0, 0, 'Z', 'Z', 'J', 0], [0, 0, 'I', 0, 0, 0, 'Z', 'J', 'J', 0], [0, 0, 'I', 0, 0, 'T', 0, 'J', 0, 0], [0, 0, 'I', 0, 'L', 'T', 'T', 'J', 0, 0], [0, 0, 'L', 'L', 'L', 'T', 'J', 'J', 0, 0], [0, 0, 0, 0, 'O', 'O', 0, 'Z', 'Z', 0], [0, 0, 0, 0, 'O', 'O', 0, 0, 'Z', 'Z'], [0, 0, 0, 0, 0, 'I', 'I', 'I', 'I', 0], [0, 0, 0, 0, 0, 0, 'S', 'S', 0, 0], [0, 0, 0, 0, 0, 'S', 'S', 0, 0, 0], [0, 0, 0, 0, 0, 0, 'L', 'L', 'L', 0], ['I', 'I', 'I', 'I', 0, 0, 'L', 'J', 'J', 'J']]
    game = Game(c.ruleset)
    game.setup()
    game.players[0].board.grid = grid
    game.players[0].held_piece = "J"
    moves = get_move_matrix(game.players[0], algo='brute-force')
    moves = get_move_list(moves, np.ones(shape=POLICY_SHAPE))
    print(moves)

def plot_stats(include_rank_data=True):
    c = Config()
    # stats_paths = [f"{c.data_dir}/stats.txt"]
    stats_paths = [f"{Config(data_version=2.3).data_dir}/stats.txt",
                   f"{Config(data_version=2.4).data_dir}/stats.txt",]

    data = {}

    for stats_path in stats_paths:
        with open(stats_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip("\n")
                line = line.split(": ", 1)[1]
                dicts = ast.literal_eval(line)
                for stat in dicts:
                    if stat not in data:
                        data[stat] = [dicts[stat]]
                    else:
                        data[stat].append(dicts[stat])

    df = pd.DataFrame(data)
    df = df.groupby("model_number").mean()

    rank_data = {
        "D": {"app": 0.154, "dspp": 0.034, "color": "#8f7591"},
        "C": {"app": 0.235, "dspp": 0.058, "color": "#733d8e"},
        "B": {"app": 0.290, "dspp": 0.081, "color": "#4f64c9"},
        "A": {"app": 0.360, "dspp": 0.105, "color": "#47ac52"},
        # "S-": {"app": 0.426, "dspp": 0.122, "color": "#b1972a"},
        "S": {"app": 0.467, "dspp": 0.133, "color": "#e1a71c"},
        # "S+": {"app": 0.517, "dspp": 0.141, "color": "#d9af0d"},
        "SS": {"app": 0.586, "dspp": 0.154, "color": "#db8a1e"},
        "U": {"app": 0.673, "dspp": 0.169, "color": "#ff3913"},
        # "X": {"app": 0.757, "dspp": 0.174, "color": "#ff45ff"},
        # "X+": {"app": 0.849, "dspp": 0.177, "color": "#653c8d"},
    }

    change_data = {
        72: "Kicktable was corrected from SRS-X to SRS+ and all-spins were fixed for the AI"
    }

    fig, axs = plt.subplots(len(df.columns), figsize=(6.5, 5.6 + 1.2 * len(df.columns)))
    fig.suptitle('Selfplay Data Statistics')

    # Collect all legend handles and labels
    all_handles = []
    all_labels = []
    rank_added = False
    change_added = False

    for i, stat in enumerate(df.columns):
        # Plot main data
        line = axs[i].plot(df[stat], label=f'{stat} data')[0]
        axs[i].set_xlabel("Model number")
        axs[i].set_ylabel(f"{stat}")


        # Shrink plot width to make space for legend
        box = axs[i].get_position()
        axs[i].set_position([box.x0, box.y0, box.width * 0.98, box.height])
        
        # Add rank reference lines
        if include_rank_data and stat in rank_data["D"]:
            for rank in rank_data:
                if stat in rank_data[rank]:
                    rank_line = axs[i].axhline(
                        y=rank_data[rank][stat], 
                        color=rank_data[rank]["color"], 
                        alpha=0.6, 
                        linestyle='-', 
                        linewidth=1
                    )
                    # Only add to legend once
                    if not rank_added:
                        all_handles.append(rank_line)
                        all_labels.append(f'Rank {rank}')
            rank_added = True
        
        # Add change indicators
        for model_number, change in change_data.items():
            change_line = axs[i].axvline(
                x=model_number-0.5, # -0.5 to put the line inbetween models
                color='red', 
                linestyle='--', 
                alpha=0.5,
                linewidth=1
            )
            # Only add to legend once
            if not change_added:
                all_handles.append(change_line)
                all_labels.append('Change')
            change_added = True

    # Create a single legend for the entire figure
    legend = None
    if all_handles:
        legend = fig.legend(
            all_handles, 
            all_labels, 
            loc='upper right', 
            bbox_to_anchor=(0.98, 0.98),
            fontsize=9,
            frameon=True,
            fancybox=True,
            shadow=True
        )

    # # Adjust layout to prevent overlap
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.93)

    # Add changes text underneath the plot
    if change_data:
        # Create text for changes
        change_text = "Changes:\n" + "\n".join([f"• Model {k}: {v}" for k, v in change_data.items()])
        
        fig.text(0.02, 0.01, change_text, 
                fontsize=7, 
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', alpha=0.8, edgecolor='blue'),
                wrap=True)

    # plt.show()

    plt.savefig(f"{directory_path}/self_play_data_statistics_{c.ruleset}_{c.data_version}.png", bbox_extra_artists=(legend,), bbox_inches='tight')
    print("Saved")


if __name__ == "__main__":

    c=Config(move_algorithm='convolutional')

    # keras.utils.set_random_seed(937)
    
    ##### Setting learning rate DOES NOT WORK

    # plot_dirichlet_noise()
    # view_visit_count_and_policy_with_and_without_dirichlet_noise()
    # profile_game()
    # test_reflected_policy()
    # visualize_policy()
    plot_stats(include_rank_data=True)

    # visualize_high_depth_replay(get_interference_network(c, load_best_model(c)), max_iter=16000)

    # visualize_get_move_matrix(c, util_move_algo_board_2)

    # test_network_versions(132, 122)
    
    # data = load_data(c, last_n_sets=20)
    # c1 = Config(epochs=5)
    # c2 = Config(epochs=5, dropout=0.75)
    # test_configs([c1, c2], 200, data=data, load_from_best_model=False)

    # instantiate_network(c, show_summary=True, save_network=False, plot_model=True)

    # test_algorithm_accuracy(truth_algo='brute-force', test_algo='convolutional')
    # time_move_matrix(algo='convolutional')

# Command for running python files
# This is for running many tests at the same time
"/Users/matthewlee/Documents/Code/Tetris Game/SRC/.venv/bin/python" "/Users/matthewlee/Documents/Code/Tetris Game/src/util.py"