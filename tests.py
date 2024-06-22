from ai import Config, get_interpreter, load_best_model, MCTS
from game import Game

from collections import namedtuple

def test_dirichlet_noise():
    # Finding different values of dirichlet alpha affect piece decisions
    alpha_values = [1.0, 0.5]
    alpha_values = {alpha: {'n_same': 0, 'n_total': 0} for alpha in alpha_values}

    model = load_best_model()
    interpreter = get_interpreter(model)

    default_config = Config()

    game = Game()
    game.setup()

    for _ in range(5):

        default_move = MCTS(default_config, game, interpreter, add_noise=False)

        for alpha_value in alpha_values:
            config = Config(DIRICHLET_ALPHA=alpha_value)
            move = MCTS(config, game, interpreter, add_noise=True)

            if move == default_move:
                alpha_values[alpha_value]['n_same'] += 1
            
            alpha_values[alpha_value]['n_total'] += 1

        # To change the board, make the default
        game.make_move(default_move)

    return alpha_values

print(test_dirichlet_noise())