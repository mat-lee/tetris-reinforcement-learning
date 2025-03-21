from ai import *
from const import *
from game import Game
from player import Player
from mover import Mover

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pygame
import time

# ------------------------- Internal Functions -------------------------
def battle_networks_win_loss(NN_1, config_1, NN_2, config_2, games, network_1_title='Network 1', network_2_title='Network 2', show_game=False, screen=None) -> list[int, int]:
    # Battle two AI's with different networks, and returns the wins and losses for each network
    wins = np.zeros((2), dtype=int)
    flip_color = False

    if config_1.ruleset != config_2.ruleset:
        raise NotImplementedError("Ruleset's aren't equal")

    for i in range(games):
        if show_game == True:
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
            if not flip_color:
                if game.turn == 0:
                    move, _, _ = MCTS(config_1, game, NN_1)    
                elif game.turn == 1:
                    move, _, _ = MCTS(config_2, game, NN_2)
            else:
                if game.turn == 1:
                    move, _, _ = MCTS(config_1, game, NN_1)    
                elif game.turn == 0:
                    move, _, _ = MCTS(config_2, game, NN_2)
            game.make_move(move)

            if show_game == True:
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

    print(network_1_title, wins, network_2_title)
    return wins

def battle_royale(interpreters, configs, names, num_games, visual=True) -> dict:
    screen = None
    if visual == True:
        screen = pygame.display.set_mode( (WIDTH, HEIGHT))

    scores={name: {} for name in names}

    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
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

def make_piece_starting_row():
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

def test_dirichlet_noise() -> None:
    # Finding different values of dirichlet alpha affect piece decisions
    alpha_values = [0.03, 0.01, 0.003, 0.001, 0.0003]
    # alpha_values = [0.4, 0.3, 0.2, 0.1]
    # alpha_values = [0.0001]
    alpha_values = {alpha: {'n_same': 0, 'n_total': 0} for alpha in alpha_values}

    use_dirichlet_s=False

    default_config = Config()

    model = load_best_model(default_config)
    interpreter = get_interpreter(model)

    for _ in range(10):
        game = Game(default_config.ruleset)
        game.setup()

        while game.is_terminal == False:
            default_move, _, _ = MCTS(default_config, game, interpreter)

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
        ax.set_xlabel(f"Dirichlet Alpha Values; Dirichlet S {default_config.DIRICHLET_S}")
        ax.set_ylabel("% of moves that were the same as without noise")

        plt.savefig(f"{directory_path}/tst_alpha_vals_s")
    else:
        ax.set_xlabel("Dirichlet Alpha Values")
        ax.set_ylabel("% of moves that were the same as without noise")

        plt.savefig(f"{directory_path}/tst_alpha_vals")

    return percent_dict

def view_policy_with_dirichlet_noise() -> None:
    # Creates a graph of the policy distribution before and after dirichlet noise is applied
    c = Config(MAX_ITER=1, 
               training=True, 
               use_playout_cap_randomization=False, 
               use_forced_playouts_and_policy_target_pruning=False, 
               use_dirichlet_noise=False)

    no_noise_config = c.copy()
    noisy_config = c.copy()
    noisy_config.use_dirichlet_noise = True

    g = Game(c.ruleset)
    g.setup()

    interpreter = get_interpreter(load_best_model(c))
    _, no_noise_tree, _ = MCTS(no_noise_config, g, interpreter)
    _, noisy_tree, _ = MCTS(noisy_config, g, interpreter)

    pre_noise_policy = get_attribute_list_from_tree(no_noise_tree, "policy")
    post_noise_policy = get_attribute_list_from_tree(noisy_tree, "policy")

    fig, axs = plt.subplots(2)
    fig.suptitle('Policy before and after dirichlet noise')
    axs[0].plot(pre_noise_policy)
    axs[1].plot(post_noise_policy)
    plt.savefig(f"{directory_path}/policy_vs_noise_{c.ruleset}_{MODEL_VERSION}.png")
    print("Saved")

def view_policy_vs_visit_count() -> None:
    # Creates a graph of the policy distribution before and after dirichlet noise is applied
    c = Config(MAX_ITER=1600,
               use_playout_cap_randomization=False, 
               use_forced_playouts_and_policy_target_pruning=False)

    g = Game(c.ruleset)
    g.setup()

    interpreter = get_interpreter(load_best_model(c))
    _, tree, _ = MCTS(c, g, interpreter)

    root_child_n = get_attribute_list_from_tree(tree, "visit_count")
    root_child_policy = get_attribute_list_from_tree(tree, "policy")

    fig, axs = plt.subplots(2)
    fig.suptitle('N Compared to policy')
    axs[0].plot(root_child_n)
    axs[1].plot(root_child_policy)
    plt.savefig(f"{directory_path}/policy_vs_n_{c.ruleset}_{MODEL_VERSION}.png")
    print("Saved")

def time_move_matrix(algo) -> None:
    # Test the game speed
    # Returns the average speed of each move over n games

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

    config = Config(MAX_ITER=100)

    num_games = 3

    # Initialize pygame
    screen = pygame.display.set_mode( (WIDTH, HEIGHT))
    pygame.display.set_caption(f'Profiling Get Move Matrix')

    interpreter = get_interpreter(load_best_model(config))

    moves = 0
    START = time.time()

    for _ in range(num_games):
        game = Game(config.ruleset)
        game.setup()

        while game.is_terminal == False:
            move, _, _ = MCTS(config, game, interpreter, move_algorithm=algo)
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

        game = Game(config.ruleset)
        game.setup()

        START = time.time()
        MCTS(config, game, network)
        END = time.time()

        scores[str(value)] = END - START
    print(scores)

def profile_game() -> None:
    game = Game(Config().ruleset)
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

    config = Config(MAX_ITER=16, move_algo='brute-force')

    interpreter = get_interpreter(load_best_model(config))

    truth_moves = 0
    test_moves = 0

    for _ in range(num_games):
        game = Game(config.ruleset)
        game.setup()

        while game.is_terminal == False:
            truth_moves += np.sum(get_move_matrix(game.players[game.turn], algo=truth_algo))
            test_moves += np.sum(get_move_matrix(game.players[game.turn], algo=test_algo))

            move, _, _ = MCTS(config, game, interpreter)
            game.make_move(move)

            game.show(screen)
            pygame.event.get()
            pygame.display.update()
    
    print(test_moves / truth_moves * 100)

def visualize_piece_placements(game=None, moves=None):
    if moves == None:
        game = Game(Config().ruleset)
        game.setup()

        # Place a piece to make it more interesting
        for i in range(10):
            piece = game.players[game.turn].piece
            game.make_move((policy_piece_to_index[piece.type][0], 0, piece.x_0, game.players[game.turn].ghost_y))
        
        game.players[0].board.grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 'Z', 'Z', 0, 0, 0],
[0, 0, 0, 0, 0, 0, 'Z', 'Z', 0, 0],
[0, 0, 0, 0, 0, 'S', 'S', 0, 0, 0],
[0, 0, 0, 0, 'S', 'S', 0, 0, 0, 0],
[0, 0, 0, 0, 0, 'I', 'I', 'I', 'I', 0],
[0, 0, 'O', 'O', 0, 0, 0, 0, 'Z', 0],
['J', 0, 'O', 'O', 0, 'L', 0, 'Z', 'Z', 0],
['J', 'J', 'J', 'L', 'L', 'L', 0, 'Z', 0, 0]]

        game.players[0].queue.pieces = ["T"]
        game.players[0].create_piece("J")

        move_matrix = get_move_matrix(game.players[game.turn], algo='brute-force')
        moves = get_move_list(move_matrix, np.ones(shape=POLICY_SHAPE))

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Tetris')
    # Need to iterate through pygame events to initialize screen
    pygame.event.get()

    moves_0 = [move for move in moves if move[1][1] == 0]
    moves_1 = [move for move in moves if move[1][1] == 1]

    for _, move in moves_0:
        game_copy = game.copy()
        game_copy.make_move(move)

        game_copy.show(screen)
        pygame.display.update()

        time.sleep(0.3)
        print("z")
    
    time.sleep(1.5)

    for _, move in moves_1:
        game_copy = game.copy()
        game_copy.make_move(move)

        game_copy.show(screen)
        pygame.display.update()

        time.sleep(1.0)
        print("o")

def test_reflected_policy():
    # Testing if reflecting pieces, grids, and policy are accurate
    game = Game(Config().ruleset)
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

    test_configs(configs, num_games, data=data, load_from_best_model=load_from_best_model, visual=visual)

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
            load_data_and_train_model(config, network, data)

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
    raise Exception("Not up to date!")
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
    print("CHECK CODE!!!")
    # raise Exception("Not up to date!")
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

def test_older_vs_newer_networks(old_ver, new_ver):
    # Making sure that the newest iteration of a network is better than earlier versions
    old_path = f"{directory_path}/models/{MODEL_VERSION}/{old_ver}.keras"
    old_model = get_interpreter(keras.models.load_model(old_path))

    new_path = f"{directory_path}/models/{MODEL_VERSION}/{new_ver}.keras"
    new_model = get_interpreter(keras.models.load_model(new_path))

    config = Config()
    screen = pygame.display.set_mode( (WIDTH, HEIGHT))
    battle_networks_win_loss(new_model, config, old_model, config, 200, f"new {new_ver}", f"old {old_ver}", True, screen)

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


def test_high_depth_replay(network, max_iter):
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

def test_convert_data_and_train_4_7_to_4_8():
    config = Config(epochs=2)
    new_network = instantiate_network(config, show_summary=True, save_network=False)

    filenames = get_data_filenames(200, shuffle=False)

    path = f"{directory_path}/data/{DATA_VERSION}"

    i = 0

    for filename in filenames:
        i += 1
        set = ujson.load(open(f"{path}/{filename}", 'r'))
        # Manipulate the set
        '''
            grids[0], pieces[0], b2b[0], combo[0], lines_cleared[0], lines_sent[0], 
            grids[1], pieces[1], b2b[1], combo[1], lines_cleared[1], lines_sent[1], 
            color, pieces_placed

            to

            grids[0], pieces[0], b2b[0], combo[0], garbage[0],
            grids[1], pieces[1], b2b[1], combo[1], garbage[1],
            color
        '''

        for move in set:
            move[4] = 0
            move[10] = 0
            move.pop(5)
            move.pop(10)
            move.pop(11)
    
        # Train challenger network
        if config.model == 'keras':
            train_network_keras(config, new_network, set)
        elif config.model == 'pytorch':
            train_network_pytorch(config, new_network, set)
        
        if i % 10 == 0:
            new_network.save(f"{directory_path}/models/TESTS/{i}.keras")
        
    new_network.save(f"{directory_path}/models/TESTS/{i}.keras")

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

    fig, axs = plt.subplots(1, 19, figsize=(25, 3))
    fig.suptitle('Policy visualization', y=0.98)
    for i in range(len(policy_index_to_piece)):
        axs[i].imshow(policy[i], cmap='viridis')
        axs[i].set_title(f"{policy_index_to_piece[i][0]} rotation {policy_index_to_piece[i][1]}")

    plt.savefig(f"{directory_path}/policy_visualization.png")
    print("saved")

def generate_human_data():
    # Manually play a game against the AI, once as white and once as black
    c = Config()
    network = get_interpreter(load_best_model(c))

    screen = pygame.display.set_mode( (WIDTH, HEIGHT))
    pygame.display.set_caption(f'Generating human data game')
    pygame.event.get()

    mover = Mover()

    # Initialize data storage
    # Each player's move data will be stored in their respective list
    total_data = []
    
    for color in [0, 1] * 10:
        game = Game(c.ruleset)
        game.setup()

        human_player = game.players[color]
        ai_player = game.players[1 - color]

        # Wipe game_data
        game_data = [[], []]

        while game.is_terminal == False and len(game.history.states) < MAX_MOVES:
            if game.turn == color:
                for event in pygame.event.get():
                    if human_player.piece != None:
                        # On key down
                        if event.type == pygame.KEYDOWN:
                            if event.key == k_move_left:
                                human_player.move_left()
                                mover.start_left()
                            elif event.key == k_move_right:
                                human_player.move_right()
                                mover.start_right()
                            elif event.key == k_soft_drop:
                                human_player.move_down()
                                mover.start_down()
                            elif event.key == k_hard_drop:
                                move_data = [*game_to_X(game)]
                                search_matrix = np.zeros(POLICY_SHAPE, dtype=int).tolist()
                                piece = game.players[color].piece
                                x = piece.x_0
                                y = piece.y_0
                                o = piece.rotation

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

                                search_matrix[policy_index][new_row][new_col + 2] = 1 # Account for buffer
                                add_search_and_move_data(c, game_data, search_matrix, move_data, game.turn)
                                
                                game.place()
                            elif event.key == k_rotate_cw:
                                human_player.try_wallkick(1)
                            elif event.key == k_rotate_180:
                                human_player.try_wallkick(2)
                            elif event.key == k_rotate_ccw:
                                human_player.try_wallkick(3)
                            elif event.key == k_hold:
                                human_player.hold_piece()

                        # On key release
                        elif event.type == pygame.KEYUP:
                            # pain
                            if event.key == k_move_left:
                                mover.stop_left()
                            elif event.key == k_move_right:
                                mover.stop_right()
                            elif event.key == k_soft_drop:
                                mover.stop_down()

                # DAS, ARR, and Softdrop
                current_time = time.time()

                # This makes das limited by FPS
                if mover.can_lr_das:
                    if mover.lr_das_start_time != None:
                        if current_time - mover.lr_das_start_time > mover.lr_das_counter:
                            if mover.lr_das_direction == "L":
                                human_player.move_left()
                            elif mover.lr_das_direction == "R":
                                human_player.move_right()      
                            mover.lr_das_counter += ARR/1000

                if mover.can_sd_das:
                    if mover.sd_start_time != None:
                        if current_time - mover.sd_start_time > mover.sd_counter:
                            human_player.move_down()
                            mover.sd_counter += (1 / SDF) / 1000

            # AI's turn
            else: 
                move, tree, save = MCTS(c, game, network)
                search_matrix = search_statistics(tree) # Moves that the network looked at
                
                # Get data
                move_data = [*game_to_X(game)]
                
                # Combines the search data and the move data to the game_data
                add_search_and_move_data(c, game_data, search_matrix, move_data, game.turn)

                game.make_move(move)

            game.show(screen)
            pygame.display.update()

        # After game ends update value
        winner = game.winner
        for player_idx in range(len(game_data)):
            if winner == -1: # Draw
                value = (0 if c.use_tanh else 0.5)
            elif winner == player_idx:
                value = 1
            else:
                value = (-1 if c.use_tanh else 0)
            # Insert value before policy for each move of that player
            for move_idx in range(len(game_data[player_idx])):
                game_data[player_idx][move_idx].insert(-1, value)

        # Reformat data to stack all moves into one continuous list
        total_data.extend(game_data[0])
        total_data.extend(game_data[1])

    json_data = ujson.dumps(total_data)

    with open(f"{directory_path}/player_data.txt", 'w') as out_file:
        out_file.write(json_data)
    
def test_data(config, data):
    model = load_best_model(Config())
    untrained_interpreter = get_interpreter(model)

    load_data_and_train_model(config, model, data)
    trained_interpreter = get_interpreter(model)

    battle_networks_win_loss(untrained_interpreter, config, trained_interpreter, config, 200, "untrained", "trained", True)

def test_quantization(config):
    # Tests if quantizing a model changes its performance
    model = load_best_model(Config())

    path = f"{directory_path}/TEMP_MODELS/savedmodel"

    # UNQUANTIZED MODEL
    model.export(path)

    uq_converter = tf.lite.TFLiteConverter.from_saved_model(path)

    uq_converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]

    # uq_converter.optimizations = [tf.lite.Optimize.DEFAULT] # NO QUANTIZATION

    uq_tflite_model = uq_converter.convert()

    uq_interpreter = tf.lite.Interpreter(model_content=uq_tflite_model)
    uq_interpreter.allocate_tensors()

    # QUANTIZED MODEL
    q_converter = tf.lite.TFLiteConverter.from_saved_model(path)

    q_converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]

    q_converter.optimizations = [tf.lite.Optimize.DEFAULT] # Quantization

    q_tflite_model = q_converter.convert()

    q_interpreter = tf.lite.Interpreter(model_content=q_tflite_model)
    q_interpreter.allocate_tensors()

    battle_networks_win_loss(uq_interpreter, config, q_interpreter, config, 200, "unquantized", "quantized", True)

if __name__ == "__main__":
    c=Config()

    # keras.utils.set_random_seed(937)

    # data = load_data(c)

    #### Setting learning rate DOES NOT WORK

    # test_architectures(DefaultConfig, nn_gens=[gen_alphasame_nn, test_10], data=data, num_games=200, visual=True)

    # test_parameters("dropout", values=[0.25, 0.4], num_games=200, data=data, load_from_best_model=False, visual=True)
    # test_configs([Config(default_model=test_8, l2_reg=1e-2), Config(default_model=test_8, l2_reg=1e-3)], num_games=200, data=data, load_from_best_model=False, visual=True)

    # test_data_parameters("augment_data", [True, False], 0.005, 1, 100, 200, load_from_best_model=True, visual=True)
    # test_parameters("learning_rate", [1e-3, 1e-2], num_games=200, data=data, load_from_best_model=True, visual=True)
    # test_parameters("loss_weights", [[1, 0.02], [0, 1]], num_games=200, data=data, load_from_best_model=False, visual=True)
    # test_data_parameters("use_experimental_features", [True, False], 1e-3, 1, 100, 200, True, True)
    # test_data_parameters("save_all", [True, False], 1e-1, 1, 100, 200, load_from_best_model=True, visual=True)

    # test_data_parameters("CPUCT", [0.75, 7.5], 0.001, 2, 100, 200, load_from_best_model=False, visual=True)
    # test_data_parameters("DIRICHLET_S", [25, 2500], 0.1, 1, 50, 100, load_from_best_model=True, visual=True)
    # test_parameters("FpuValue", [0.2, 0.4], num_games=500, load_from_best_model=True, visual=True)
    # test_data_parameters("FpuValue", [0.2, 0.4], 0.1, 1, 100, 200, load_from_best_model=True, visual=True)

    # test_reflected_policy()

    # test_algorithm_accuracy(test_algo='faster-but-loss')
    # time_move_matrix('faster-but-loss')


    # test_dirichlet_noise()
    # test_older_vs_newer_networks(14, 28)


    # test_high_depth_replay(get_interpreter(load_best_model(c)), max_iter=1600)
    # test_convert_data_and_train_4_7_to_4_8()

    # visualize_piece_placements()
    # test_dirichlet_noise()
    # test_parameters("FpuStrategy", ['reduction', 'absolute'], num_games=200, data=data, load_from_best_model=True, visual=True)

    # visualize_policy()

    # play_game(c, get_interpreter(load_best_model(c)), 777, show_game=True)
    # generate_human_data()

    # data = [ujson.load(open(f"{directory_path}/player_data.txt", 'r'))]
    # test_data(c, data)

    # view_policy_with_dirichlet_noise()
    # view_policy_vs_visit_count()

    test_quantization(c)

    # Command for running python files
    # This is for running many tests at the same time
    "/Users/matthewlee/Documents/Code/Tetris Game/SRC/.venv/bin/python" "/Users/matthewlee/Documents/Code/Tetris Game/src/tests.py"