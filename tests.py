from ai import *
from const import *
from game import Game

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pygame
import time

# ------------------------- Internal Functions -------------------------
def battle_networks_win_loss(NN_1, config_1, NN_2, config_2, games, network_1_title='Network 1', network_2_title='Network 2', show_game=False, screen=None) -> list[int, int]:
    # Battle two AI's with different networks, and returns the wins and losses for each network
    wins = np.zeros((2), dtype=int)
    for i in range(games):
        if show_game == True:
            if screen != None:
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

    print(network_1_title, wins, network_2_title)
    return wins

def battle_royale(interpreters, configs, names, num_games, visual=True) -> dict:
    screen = None
    if visual == True:
        screen = pygame.display.set_mode( (WIDTH, HEIGHT))

    scores={name: {} for name in names}

    for i in range(len(interpreters)):
        for j in range(i):
            if i != j:
                score_1, score_2 = battle_networks_win_loss(interpreters[i], configs[i], 
                                                            interpreters[j], configs[j],
                                                            num_games, 
                                                            network_1_title=names[i], 
                                                            network_2_title=names[j], 
                                                            show_game=visual, screen=screen)
                
                scores[names[i]][names[j]] = f"{score_1}-{score_2}"
                scores[names[j]][names[i]] = f"{score_2}-{score_1}"

    return scores

# ------------------------- Test Functions -------------------------

def test_dirichlet_noise() -> None:
    # Finding different values of dirichlet alpha affect piece decisions
    alpha_values = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    alpha_values = {alpha: {'n_same': 0, 'n_total': 0} for alpha in alpha_values}

    model = load_best_model()
    interpreter = get_interpreter(model)

    default_config = Config()

    for _ in range(100):
        game = Game()
        game.setup()

        for _ in range(10):
            default_move, _, _ = MCTS(default_config, game, interpreter, add_noise=False)

            for alpha_value in alpha_values:
                config = Config(DIRICHLET_ALPHA=alpha_value)
                move, _, _ = MCTS(config, game, interpreter, add_noise=True)

                if move == default_move:
                    alpha_values[alpha_value]['n_same'] += 1
                
                alpha_values[alpha_value]['n_total'] += 1

            # To change the board, make the default move
            game.make_move(default_move)

    percent_dict = {alpha: 100*alpha_values[alpha]['n_same']/alpha_values[alpha]['n_total'] for alpha in alpha_values}    

    fig, ax = plt.subplots()

    ax.bar(range(len(percent_dict)), list(percent_dict.values()), align='center')
    ax.set_xticks(range(len(percent_dict)), list(percent_dict.keys()))

    ax.set_xlabel("Dirichlet Alpha Values")
    ax.set_ylabel("% of moves that were the same as without noise")

    plt.savefig(f"{directory_path}/tst_alpha_vals")

    return percent_dict

def time_move_matrix() -> None:
    # Test the game speed
    # Returns the average speed of each move over n games

    # ---------- 100 iter ----------
    # Initial:                        0.340 0.357 0.362
    # Deque:                          0.382
    # Deque + set:                    0.310
    # Pop first:                      0.320
    # Don't use array                 0.297
        # Using mp Queue              0.354
    
    # Default                         0.298
    # Don't check o rotations         0.280
    # Use single softdrop             0.339
    # Using fast algo                 0.194

    num_games = 10

    # Initialize pygame
    screen = pygame.display.set_mode( (WIDTH, HEIGHT))
    pygame.display.set_caption(f'Profiling Get Move Matrix')

    interpreter = get_interpreter(load_best_model())

    config = Config(MAX_ITER=100)

    moves = 0
    START = time.time()

    for _ in range(num_games):
        game = Game()
        game.setup()

        while game.is_terminal == False:
            move, _, _ = MCTS(config, game, interpreter)
            game.make_move(move)
            moves += 1

            game.show(screen)
            pygame.event.get()
            pygame.display.update()

    END = time.time()

    print((END-START)/moves)

def time_architectures(var, values) -> None:
    # Tests how fast an algorithm takes to make a move
    scores = {}
    configs = [Config(MAX_ITER=100) for _ in range(len(values))]
    for value, config in zip(values, configs):
        setattr(config, var, value)
        network = get_interpreter(instantiate_network(config,show_summary=False, save_network=False))

        game = Game()
        game.setup()

        START = time.time()
        MCTS(config, game, network)
        END = time.time()

        scores[str(value)] = END - START
    print(scores)

def profile_game() -> None:
    game = Game()
    game.setup()

    config = Config()
    network = get_interpreter(load_best_model())

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Testing algorithm accuracy')

    # Profile game
    with cProfile.Profile() as pr:
        play_game(config, network, 777, show_game=True, screen=screen)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(20)

def test_algorithm_accuracy(truth_algo='brute-force', test_algo='faster-but-loss') -> None:
    # Test how accurate an algorithm is
    # Returns the percent of moves an algorithm found compared to all possible moves

    num_games = 10

    # Initialize pygame
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Testing algorithm accuracy')

    interpreter = get_interpreter(load_best_model())

    config = Config()

    truth_moves = 0
    test_moves = 0

    for _ in range(num_games):
        game = Game()
        game.setup()

        while game.is_terminal == False:
            truth_moves += np.sum(get_move_matrix(game.players[game.turn], algo=truth_algo))
            test_moves += np.sum(get_move_matrix(game.players[game.turn], algo=test_algo))

            move, _, _ = MCTS(config, game, interpreter, move_algorithm='brute-force')
            game.make_move(move)

            game.show(screen)
            pygame.event.get()
            pygame.display.update()
    
    print(test_moves / truth_moves * 100)

def test_reflected_policy():
    # Testing if reflecting pieces, grids, and policy are accurate
    def visualize_piece_placements(game, moves):
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

    move_matrix = get_move_matrix(game.players[game.turn], algo='brute-force')
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

def test_parameters(
    var: str, 
    values: list,
    num_games: int,
    data=None,
    load_from_best_model: bool=False,
    visual: bool=True
):
    ## Grid search battling different parameters
    # Configs
    configs = [Config() for _ in range(len(values))]
    for value, config in zip(values, configs):
        setattr(config, var, value)

    # Networks
    if load_from_best_model:
        networks = [load_best_model(config) for config in configs]
    else:
        networks = [instantiate_network(config, show_summary=False, save_network=False, plot_model=False) for config in configs]

    if data != None:
        for config, network in zip(configs, networks):
            for set in data:
                train_network_keras(config, network, set)
        
        del data
        gc.collect()

    # Networks -> interpreters
    interpreters = [get_interpreter(network) for network in networks]

    print(battle_royale(interpreters, 
                        configs, 
                        [str(value) for value in values], 
                        num_games,
                        visual=visual))

def test_configs(
    configs,
    num_games: int,
    data=None,
    load_from_best_model: bool=False,
    visual: bool=True
):
    ## Grid search battling different Configs

    # Networks
    if load_from_best_model:
        networks = [load_best_model(config) for config in configs]
    else:
        networks = [instantiate_network(config, show_summary=False, save_network=False, plot_model=False) for config in configs]

    if data != None:
        for config, network in zip(configs, networks):
            for set in data:
                train_network_keras(config, network, set)
        
        del data
        gc.collect()

    # Networks -> interpreters
    interpreters = [get_interpreter(network) for network in networks]

    print(battle_royale(interpreters, 
                        configs, 
                        [f"Config {i+1}" for i in range(len(configs))], 
                        num_games,
                        visual=visual))

def test_architectures(
    config,
    nn_gens: list, 
    data,
    num_games,
    visual=True
):
    # Configs aren't being changed here but the list is needed for battle royale
    configs = [copy.deepcopy(config) for _ in range(len(nn_gens))]
    
    networks = [instantiate_network(configs[0], 
                                    nn_generator=nn_gen, 
                                    show_summary=True, 
                                    save_network=False, 
                                    plot_model=False) for nn_gen in nn_gens]

    for network in networks:
        for set in data:
            if config.model == 'keras':
                train_network_keras(configs[0], network, set)
            elif config.model == 'pytorch':
                raise NotImplementedError

    del data
    gc.collect()

    interpreters = [get_interpreter(network) for network in networks]

    print(battle_royale(interpreters, 
                        configs, 
                        [str(nn_gen) for nn_gen in nn_gens],
                        num_games=num_games, 
                        visual=visual))

def test_data_parameters(
    var: str, 
    values: list,
    learning_rate: float,
    num_training_loops: int,
    num_training_games: int,
    num_battle_games: int,
    load_from_best_model: bool = False,
    visual = True
):
    ## Grid search battling different parameters
    screen = None
    if visual == True:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
    
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
            set = make_training_set(config, interpreter, num_training_games, save_game=False, show_game=visual, screen=screen)
            
            train_network_keras(config, network, set)

            del set
            gc.collect()

    # Networks -> interpreters
    interpreters = [get_interpreter(network) for network in networks]

    # When battling, have each use the same config
    battle_configs = [Config() for _ in range(len(values))]

    print(battle_royale(interpreters, 
                        battle_configs, 
                        [str(value) for value in values], 
                        num_battle_games,
                        visual=visual))
    
    print(var)

def test_older_vs_newer_networks():
    # Making sure that the newest iteration of a network is better than earlier versions
    best_model = get_interpreter(load_best_model())

    path = f"{directory_path}/models/{MODEL_VERSION}.0.keras"
    old_model = get_interpreter(keras.models.load_model(path))

    config = Config()
    screen = pygame.display.set_mode( (WIDTH, HEIGHT))
    battle_networks_win_loss(best_model, config, old_model, config, 200, "new", "old", True, screen)

def test_if_changes_improved_model():
    config = Config()
    network = instantiate_network(config, nn_generator=None, show_summary=False, save_network=False, plot_model=False)
    data = load_data(last_n_sets=10)

    for set in data:
        train_network_keras(config, network, set)

        del set
        gc.collect()

    best_nn = get_interpreter(load_best_model(config))
    chal_nn = get_interpreter(network)

    screen = pygame.display.set_mode( (WIDTH, HEIGHT))

    if battle_networks(chal_nn, config, best_nn, config, 0.55, 'moreorequal', 200, "New", "Best", show_game=True, screen=screen):
        print("Success")
    else:
        print("Failure")
    network.save(f"{directory_path}/New.keras")

def test_old_data(data):
    # To see if the old policy size (i, r - 1, c + 1) could be converted to (i, r - 1, c)
    for set in data:
        for move in set:
            policy = move[-1]
            for idx in policy:
                for row in idx:
                    if row[-1] != 0:
                        print("e")
                        a = 'e'


c=Config(model='keras', shuffle=True, MAX_ITER=160, epochs=1)

# keras.utils.set_random_seed(937)

new_nn = instantiate_network(c, test_11, show_summary=True, save_network=False, plot_model=False)

data = load_data(data_ver=1.3, last_n_sets=1)

new_data = convert_old_policy_to_new_policy(data)

for set in new_data:
    train_network_keras(c, new_nn, set)

new_nn.save(f"{directory_path}/new.keras")

#### Setting learning rate DOES NOT WORK

# test_architectures(DefaultConfig, nn_gens=[gen_alphasame_nn, test_10], data=data, num_games=200, visual=True)

# test_parameters("dropout", values=[0.25, 0.4], num_games=200, data=data, load_from_best_model=False, visual=True)
# test_configs([Config(default_model=test_8, l2_reg=1e-2), Config(default_model=test_8, l2_reg=1e-3)], num_games=200, data=data, load_from_best_model=False, visual=True)

# test_data_parameters("augment_data", [True, False], 0.005, 1, 100, 200, load_from_best_model=True, visual=True)
# test_parameters("learning_rate", [1e-3, 1e-2], num_games=200, data=data, load_from_best_model=True, visual=True)
# test_data_parameters("use_experimental_features", [True, False], 1e-3, 1, 100, 200, True, True)
# test_data_parameters("save_all", [True, False], 1e-1, 1, 100, 200, load_from_best_model=True, visual=True)

# test_data_parameters("DIRICHLET_S", [25, 2500], 0.1, 1, 50, 100, load_from_best_model=True, visual=True)
# test_data_parameters("FpuValue", [0.1, 0.01], 0.1, 1, 100, 200, load_from_best_model=True, visual=True)

# test_reflected_policy()

# Command for running python files
# This is for running many tests at the same time
"/Users/matthewlee/Documents/Code/Tetris Game/SRC/.venv/bin/python" "/Users/matthewlee/Documents/Code/Tetris Game/src/tests.py"