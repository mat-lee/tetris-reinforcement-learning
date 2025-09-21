from ai import *
from const import *
from game import Game
from piece import Piece
from player import Player

import ast
from collections import namedtuple
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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

    #   convolution                   0.247
    #   faster-but-loss               0.244
    #   faster-conv                   0.187 94%
    #   ultra-conv                    0.159 

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

def profile_game(c) -> None:
    game = Game(c.ruleset)
    game.setup()

    network = get_interpreter(load_best_model(c))

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Testing algorithm accuracy')

    # Profile game
    with cProfile.Profile() as pr:
        play_game(c, network, 777, screen=screen)
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

def convert_data_2_4_to_2_5(set):
    # Add aux data to set
    for move in set:
        grid = move[0]
        metrics = calculate_board_metrics(grid)
        holes = metrics['holes']
        avg_height = metrics['avg_height']

        move.append(avg_height)
        move.append(holes)

def convert_data_and_train(c, init_data_ver, conversion_function, last_n_sets, epochs):
    c.epochs = epochs
    c.data_version = init_data_ver

    new_network = instantiate_network(c, show_summary=True, save_network=False)
    # new_network = load_model(c, 300)

    filenames = get_data_filenames(c, last_n_sets=last_n_sets)

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

def visualize_policy_from_data():
    # Visualize the policy of a network
    c=Config()

    data = load_data(c, last_n_sets=1)
    data = data[0][:50]

    frames = []

    for move in data:
        policy = move[-3] # value policy height holes
        fig, axs = plt.subplots(1, POLICY_SHAPE[0], figsize=(40, 3))
        fig.suptitle('Policy visualization', y=0.98)

        for i in range(len(policy_index_to_piece)):
            axs[i].imshow(policy[i], cmap='viridis')

        # Draw and convert to RGB array
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.array(canvas.buffer_rgba())[:, :, :3]  # (H, W, 3) array

        frames.append(img)
        plt.close(fig)

    iio.imwrite(f"{directory_path}/sinewave.mp4", frames, fps=10)

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

import matplotlib as mpl

# Make figures crisp by default
mpl.rcParams.update({
    "figure.dpi": 150,      # notebook / on-screen
    "savefig.dpi": 300,     # file output
    "lines.linewidth": 0.9, # default thinner lines
    "axes.linewidth": 0.8,
    "patch.antialiased": True,
    "text.antialiased": True,
})

def plot_stats(model_version=None, data_version=None, average_by_model=False, include_rank_data=True):
    """
    Plot app and dspp statistics from a single stats file for specific versions.
    """
    data = {}
    stats_path = f"{directory_path}/data/stats.txt"

    with open(stats_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            if not line:
                continue
            try:
                dicts = ast.literal_eval(line)

                # Filter by versions
                if model_version is not None:
                    if isinstance(model_version, list):
                        if dicts['model_version'] not in model_version:
                            continue
                    else:
                        if dicts['model_version'] != model_version:
                            continue

                if data_version is not None:
                    if isinstance(data_version, list):
                        if dicts['data_version'] not in data_version:
                            continue
                    else:
                        if dicts['data_version'] != data_version:
                            continue

                # Collect
                for stat in dicts:
                    data.setdefault(stat, []).append(dicts[stat])

            except Exception:
                continue

    df = pd.DataFrame(data)
    if df.empty or 'app' not in df.columns or 'dspp' not in df.columns:
        print("No valid data found with app/dspp columns")
        return None, None

    if average_by_model:
        df = df.groupby(['model_number', 'model_version'])[['app', 'dspp']].mean().reset_index()

    rank_data = {
        "D": {"app": 0.154, "dspp": 0.034, "color": "#8f7591"},
        "C": {"app": 0.235, "dspp": 0.058, "color": "#733d8e"},
        "B": {"app": 0.290, "dspp": 0.081, "color": "#4f64c9"},
        "A": {"app": 0.360, "dspp": 0.105, "color": "#47ac52"},
        "S": {"app": 0.467, "dspp": 0.133, "color": "#e1a71c"},
        "SS": {"app": 0.586, "dspp": 0.154, "color": "#db8a1e"},
        "U": {"app": 0.673, "dspp": 0.169, "color": "#ff3913"},
    }

    change_data = {
        (5.9, 2.4): "Kicktable was corrected from SRS-X to SRS+ and all-spins were fixed for the AI",
        (5.9, 2.6): "Changed temperature from 0.0 -> 0.1, CForcedPlayout 2 -> 1, DIRICHLET_ALPHA 0.02 -> 0.05, SETS_TO_TRAIN_WITH 10 -> 5"
    }

    # Slightly wider figure; constrained_layout to avoid overlap and preserve sharpness
    fig, axs = plt.subplots(2, figsize=(7.5, 6.8 + 1.2 * 2), constrained_layout=True)
    fig.suptitle('Selfplay Data Statistics')

    all_handles, all_labels = [], []
    rank_added = False
    change_added = False
    stats = ['app', 'dspp']

    for i, stat in enumerate(stats):
        # Thin line, antialiased, no markers to avoid clutter
        axs[i].plot(
            df[stat].values,
            label=f'{stat}',
            linewidth=0.9,
            antialiased=True
        )
        axs[i].set_xlabel("Model number" if average_by_model else "Training step")
        axs[i].set_ylabel(stat)

        # subtle grid to help see small changes
        axs[i].grid(True, linewidth=0.5, alpha=0.3)

        # Move plot a touch right (kept from your original intent)
        box = axs[i].get_position()
        axs[i].set_position([box.x0, box.y0, box.width * 0.985, box.height])

        if include_rank_data and stat in rank_data["D"]:
            for rank, rinfo in rank_data.items():
                if stat in rinfo:
                    rank_line = axs[i].axhline(
                        y=rinfo[stat],
                        color=rinfo["color"],
                        alpha=0.5,
                        linestyle='-',
                        linewidth=0.6,  # thinner ref lines
                        zorder=0
                    )
                    if not rank_added:
                        all_handles.append(rank_line)
                        all_labels.append(f'Rank {rank}')
            rank_added = True

        # Change indicators (thin dashed)
        for (mv, dv), change in change_data.items():
            matching = df[(df.get('model_version') == mv) & (df.get('data_version') == dv)] if {'model_version','data_version'}.issubset(df.columns) else pd.DataFrame()
            if not matching.empty:
                x_pos = (matching.index[0] if not average_by_model else matching['model_number'].iloc[0])
                change_line = axs[i].axvline(
                    x=x_pos,
                    color='red',
                    linestyle='--',
                    alpha=0.5,
                    linewidth=0.6
                )
                if not change_added:
                    all_handles.append(change_line)
                    all_labels.append('Change')
                    change_added = True

    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            fontsize=9,
            frameon=True,
            fancybox=True,
            shadow=True
        )

    if change_data:
        change_text = "Changes:\n" + "\n".join([f"• Model {mv}, Data {dv}: {desc}" for (mv, dv), desc in change_data.items()])
        fig.text(
            0.02, 0.01, change_text,
            fontsize=7,
            va='bottom',
            ha='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', alpha=0.8, edgecolor='blue'),
            wrap=True
        )

    png_path = f"{directory_path}/self_play_data_statistics_test.png"
    plt.savefig(png_path, bbox_inches='tight', facecolor='white')
    print(f"Saved {png_path}")

    return fig, axs

def migrate_stats_data():
    """
    Migrates stats data from old format to new format.
    
    Old format: data_number: {'model_number': 72, 'app': 0.109, 'dspp': 0.169}
    New format: {"model_number": model_number, "model_version": config.model_version, 
                "data_number": data_number, "data_version": config.data_version, 
                "app": app, "dspp": dspp}
    """
    
    # Only check specific data versions as requested
    data_versions = [2.3, 2.4]
    model_version = 5.9 # The model version used to make the data
    print(f"Processing data versions: {data_versions}")
    
    # Prepare output file - check if it exists to determine if we should append
    output_file = f"{directory_path}/data/stats.txt"
    
    file_mode = 'a' if os.path.exists(output_file) else 'w'
    if file_mode == 'a':
        print(f"Appending to existing {output_file}")
    else:
        print(f"Creating new {output_file}")
    
    # Process each data version
    total_records = 0
    
    with open(output_file, file_mode) as output_f:
        for data_version in data_versions:
            config = Config(data_version=data_version)
            stats_path = f"{config.data_dir}/stats.txt"
            
            print(f"Processing {stats_path}...")
            
            try:
                with open(stats_path, 'r') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                        
                    try:
                        # Parse the old format: data_number: {'model_number': 72, 'app': 0.109, 'dspp': 0.169}
                        if ": " not in line:
                            print(f"Warning: Invalid format in {stats_path} line {line_num}: {line}")
                            continue
                            
                        data_number_str, dict_str = line.split(": ", 1)
                        data_number = int(data_number_str)
                        old_data = ast.literal_eval(dict_str)
                        
                        # Create new format record
                        new_record = {
                            "model_number": old_data.get("model_number"),
                            "model_version": model_version,
                            "data_number": data_number,
                            "data_version": config.data_version,
                            "app": old_data.get("app"),
                            "dspp": old_data.get("dspp")
                        }
                        
                        # Write directly to file
                        output_f.write(ujson.dumps(new_record) + "\n")
                        total_records += 1
                        
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing line {line_num} in {stats_path}: {e}")
                        print(f"Line content: {line}")
                        continue
                        
            except FileNotFoundError:
                print(f"Error: Could not read {stats_path}")
            except Exception as e:
                print(f"Unexpected error processing {stats_path}: {e}")
    
    print(f"\nMigration complete!")
    print(f"Processed {total_records} new records")
    print(f"Data written to: {output_file}")

def test_policy_piece_invariance(c):
    c.MAX_ITER = 2
    game = Game(c.ruleset)
    game.setup()

    model_path = f"{directory_path}/models/debug/test_model.keras"
    print(f"Loading model from {model_path}")
    model = get_interpreter(keras.models.load_model(model_path))

    policies = []

    # After a certain number of moves, the policy is examined
    moves = 10

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Viewing policy of the network")
    pygame.event.get() # Required for visuals?

    while game.is_terminal == False and len(game.history.states) < moves:
        move, _, _ = MCTS(c, game, model)
        game.make_move(move)

        game.show(screen)
        pygame.display.update()

    for piece in MINOS:
        game.players[game.turn].piece = Piece(type=piece)
        game.players[game.turn].piece.move_to_spawn()
        game.players[game.turn].held_piece = piece

        value, policy = evaluate(c, game, model)

        policies.append(policy)
    
    for i in range(len(policies)):
        for j in range(i + 1, len(policies)):
            # Compute cosine similarity
            cos_sim = np.sum(policies[i] * policies[j]) / (np.linalg.norm(policies[i]) * np.linalg.norm(policies[j]))
            print(f"Cosine similarity between {MINOS[i]} and {MINOS[j]}: {cos_sim}")
    
    fig, axs = plt.subplots(len(MINOS), POLICY_SHAPE[0], figsize=(40, 3 * len(MINOS)))
    fig.suptitle('Policy visualization', y=0.98)
    for i, piece in enumerate(MINOS):
        for j in range(len(policy_index_to_piece)):
            axs[i, j].imshow(policies[i][j], cmap='viridis')

    plt.tight_layout()
    plt.savefig(f"{directory_path}/policy_piece_invariance_{c.model_version}.png")
    print("Saved figure and printed cosine similarities")

def test_simple_piece_network_piece_invariance():
    """
    Test a very simple network that only takes piece input to see if it can learn piece invariance.
    """

    # A very simple network to test piece invariance
    piece_input = keras.Input(shape=(len(MINOS) * 7,), name='piece_input')
    policy = keras.layers.Dense(POLICY_SIZE, activation='softmax')(piece_input)

    model = keras.Model(inputs=piece_input, outputs=policy)


    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Make training data and train model
    policies = []
    pieces = []
    data = load_data(Config(data_version=2.4), last_n_sets=20)
    for set in data:
        for move in set:
            policy = move[-1]
            # print(np.argwhere(np.array(policy)))
            policy = np.array(policy).reshape(POLICY_SIZE,)
            policies.append(policy)

            piece_matrix = move[1]
            piece_matrix = np.array(piece_matrix).reshape(len(MINOS) * 7,)

            pieces.append(piece_matrix) # Piece is second item, first player

    model.fit(np.array(pieces), np.array(policies), epochs=1, batch_size=32)

    # Visualize policy for each piece
    policies = []
    for piece in MINOS:
        piece_one_hot = np.zeros((7, len(MINOS)))
        piece_one_hot[0, MINOS.index(piece)] = 1

        piece_one_hot = piece_one_hot.flatten().reshape((1, len(MINOS) * 7))

        policy = model.predict(piece_one_hot)[0]
        policy = policy.reshape(POLICY_SHAPE)

        policies.append(policy)

    fig, axs = plt.subplots(len(MINOS), POLICY_SHAPE[0], figsize=(40, 3 * len(MINOS)))
    fig.suptitle('Policy visualization', y=0.98)
    for i, piece in enumerate(MINOS):
        for j in range(len(policy_index_to_piece)):
            axs[i, j].imshow(policies[i][j], cmap='viridis')
    
    plt.tight_layout()
    plt.savefig(f"{directory_path}/simple_piece_policy_piece_invariance.png")
    print("Saved figure")

if __name__ == "__main__":
    
    c = Config()

    # data = load_data(c, last_n_sets=20)

    # keras.utils.set_random_seed(937)
    
    ##### Setting learning rate DOES NOT WORK

    # plot_dirichlet_noise()
    # view_visit_count_and_policy_with_and_without_dirichlet_noise()
    # profile_game()
    # test_reflected_policy()
    # visualize_policy()
    plot_stats(data_version=2.6, include_rank_data=True)

    # visualize_high_depth_replay(get_interference_network(c, load_best_model(c)), max_iter=16000)

    # visualize_get_move_matrix(c, util_move_algo_board_2)

    # test_network_versions(132, 122)
    
    # data = load_data(c, last_n_sets=20)
    # c1 = Config(data_version=2.4, default_model=gen_model_aux)
    # c2 = Config(data_version=2.4, default_model=gen_test1)
    # test_configs([c1, c2], 200, data=data, load_from_best_model=False)

    # instantiate_network(c, show_summary=True, save_network=False, plot_model=True)

    # test_algorithm_accuracy(truth_algo='brute-force', test_algo='faster-conv')
    # time_move_matrix(algo='faster-conv')
    # profile_game(Config(move_algorithm='ultra-conv'))
    # time_move_matrix(algo='faster-but-loss')

    # visualize_policy_from_data()
    
    # convert_data_and_train(c, 2.4, convert_data_2_4_to_2_5, last_n_sets=50, epochs=1)

    # migrate_stats_data()

    # test_policy_piece_invariance(c)
    # test_simple_piece_network_piece_invariance()

    # c = Config(default_model=gen_test_model, data_version=2.4)

    # model = instantiate_network(c, show_summary=True, save_network=False)
    # load_data_and_train_model(c, model, last_n_sets=20)
    # model.save(f"{directory_path}/models/debug/test_model.keras")

# Command for running python files
# This is for running many tests at the same time
"/Users/matthewlee/Documents/Code/Tetris Game/SRC/.venv/bin/python" "/Users/matthewlee/Documents/Code/Tetris Game/src/util.py"