from ai import battle_networks_win_loss, Config, create_network, directory_path, get_interpreter, load_best_model, load_data, MCTS, train_network
from const import *
from game import Game

from collections import namedtuple
import matplotlib.pyplot as plt
import pygame
import time

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
            default_move, _ = MCTS(default_config, game, interpreter, add_noise=False)

            for alpha_value in alpha_values:
                config = Config(DIRICHLET_ALPHA=alpha_value)
                move, _ = MCTS(config, game, interpreter, add_noise=True)

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

    num_games = 10

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode( (WIDTH, HEIGHT))
    pygame.display.set_caption(f'Profiling Get Move Matrix')

    interpreter = get_interpreter(load_best_model())

    config = Config()

    moves = 0
    START = time.time()

    for _ in range(num_games):
        game = Game()
        game.setup()

        while game.is_terminal == False:
            move, _ = MCTS(config, game, interpreter)
            game.make_move(move)
            moves += 1

            game.show(screen)
            pygame.event.get()
            pygame.display.update()

    END = time.time()

    print((END-START)/moves)

def battle_parameters(load_from_best_model: bool = False,
                      data: list[list] = None, 
                      var: str = "", 
                      values: list[int] = None) -> None:
    ## Grid search battling different parameters

    configs = [Config() for _ in range(len(values))]
    for value, config in zip(values, configs):
        setattr(config, var, value)

    if load_from_best_model == False:
        networks = [create_network(config, show_summary=False, save_network=False, plot_model=False) for config in configs]

        for config, network in zip(configs, networks):
            train_network(config, network, data)
    else:
        networks = [load_best_model() for _ in range(len(values))]

    interpreters = [get_interpreter(network) for network in networks]

    scores={str(title): {} for title in values}

    pygame.init()
    screen = pygame.display.set_mode( (WIDTH, HEIGHT))

    for i in range(len(values)):
        first_network = interpreters[i]
        first_config = configs[i]
        for j in range(i):
            if i != j:
                second_network = interpreters[j]
                second_config = configs[j]
                score_1, score_2 = battle_networks_win_loss(first_network, first_config, 
                                                            second_network, second_config,
                                                            100, 
                                                            network_1_title=values[i], 
                                                            network_2_title=values[j], 
                                                            show_game=True, screen=screen)
                
                scores[str(values[i])][str(values[j])] = f"{score_1}-{score_2}"
                scores[str(values[j])][str(values[i])] = f"{score_2}-{score_1}"

    print(scores)

battle_parameters(load_from_best_model=True, var="MAX_ITER", values=[1, 800])