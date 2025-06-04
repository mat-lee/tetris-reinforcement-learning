from ai import *
from tests import battle_networks_win_loss

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

param_space = [
    Real(0.0, 0.9, name='dropout'),
    Categorical([False, True], name='use_tanh'),

    Real(1, 100, name='CPUCT'),
    Categorical(['reduction', 'absolute'], name='FpuStrategy'),
    Real(0.0, 1.0, name='FpuValue'),
    Categorical([False, True], name='use_root_softmax'),

    Real(0.0001, 0.01, name='learning_rate'),
    Real(0.0, 10.0, name='policy_loss_weight'),
    Categorical(['merge', 'distinct'], name='data_loading_style'),
    Categorical([False, True], name='augment_data'),
    Categorical([False, True], name='use_experimental_features'),
    Categorical([False, True], name='save_all'),
    Categorical([False, True], name='use_playout_cap_randomization'),
    Real(0.0, 1.0, name='dirichlet_alpha'),
    Categorical([False, True], name='use_dirichlet_s'),
    Categorical([False, True], name='use_forced_playouts_and_policy_target_pruning'),
    Integer(0.0, 3.0, name='CForcedPlayout'),
]

training_games = 10 # Number of training games per training loop
training_loops = 4 # Number of training loops
eval_games = 40 # Number of evaluation games
visual = True
screen = pygame.display.set_mode((WIDTH, HEIGHT))
i = 0

def evaluate_challenger(baseline_config, challenger_config, challenger_interpreter, num_games, visual, screen):
    # Returns the percentage of games won by the challenger network against the baseline network
    baseline_interpreter = get_interpreter(load_best_model(baseline_config))

    wins = battle_networks_win_loss(baseline_interpreter, baseline_config, challenger_interpreter, challenger_config, num_games, "Baseline Network", "Challenger Network", show_game=visual, screen=screen)
    chal_wins = wins[1]
    eval = chal_wins / num_games # Win more -> higher eval
    return eval

@use_named_args(param_space)
def objective_function(**params):
    global i # Counter
    i += 1
    print(i, params)
    baseline_config = Config()
    challenger_config = Config()

    # Set the challenger configuration parameters
    challenger_config.training = True

    for param in params:
        if param == 'policy_loss_weight':
            challenger_config.loss_weights = [1, params[param]]
        else:
            setattr(challenger_config, param, params[param])

    challenger_network = instantiate_network(challenger_config, show_summary=False, save_network=False, plot_model=False)

    # Train the challenger network
    for _ in range(training_loops):
        interpreter = get_interpreter(challenger_network)
        set = make_training_set(challenger_config, interpreter, training_games, save_game=False, show_game=visual, screen=screen)

        train_network(challenger_config, challenger_network, set)

        del set
        gc.collect()

    challenger_interpreter = get_interpreter(challenger_network)

    # Evaluate the challenge network
    result = evaluate_challenger(baseline_config, challenger_config, challenger_interpreter, eval_games, visual, screen)

    # Negate score because skopt minimizes the objective
    return -result

if __name__ == "__main__":
    # Run the optimization
    print("Starting optimization...")
    print(f"Parameter space: {param_space}")

    result = gp_minimize(
        func=objective_function,       # Objective function
        dimensions=param_space,        # Parameter space
        n_calls=1,                    # Number of evaluations
        random_state=42                # For reproducibility
    )

    print(f"Best Parameters: {result.x}")
    print(f"Best Score: {-result.fun}")  # Negate the score to interpret correctly
    print([print(x) for x in result.x_iters])