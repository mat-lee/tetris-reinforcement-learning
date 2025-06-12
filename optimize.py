from ai import *

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

param_space = [
    # Real(0.0, 0.9, name='dropout'),
    # Categorical([False, True], name='use_tanh'),

    Real(0.1, 10, name='CPUCT'),
    Real(0.001, 10, name='DPUCT'),
    # Categorical(['reduction', 'absolute'], name='FpuStrategy'),
    # Real(0.0, 1.0, name='FpuValue'),
    # Categorical([False, True], name='use_root_softmax'),

    # Real(0.0001, 0.01, name='learning_rate'),
    # Real(0.0, 10.0, name='policy_loss_weight'),
    # Categorical(['merge', 'distinct'], name='data_loading_style'),
    # Categorical([False, True], name='augment_data'),
    # Categorical([False, True], name='save_all'),
    # Categorical([False, True], name='use_playout_cap_randomization'),
    # Real(0.0, 1.0, name='DIRICHLET_ALPHA'),
    # Categorical([False, True], name='use_dirichlet_s'),
    # Categorical([False, True], name='use_forced_playouts_and_policy_target_pruning'),
    # Integer(0.0, 3.0, name='CForcedPlayout'),
]

training_games = 1 # Number of training games per training loop
training_loops = 1 # Number of training loops
eval_games = 1 # Number of evaluation games
visual = True
screen = pygame.display.set_mode((WIDTH, HEIGHT))
i = 0
baseline_model_number = 0 # The model number of the baseline network to compare against

def evaluate_challenger(baseline_config, challenger_config, challenger_interpreter, num_games, visual, screen):
    # Returns the percentage of games won by the challenger network against the baseline network
    baseline_interpreter = get_interpreter(load_model(baseline_config, baseline_model_number))

    wins, _ = battle_networks(baseline_interpreter, baseline_config, challenger_interpreter, challenger_config, None, None, num_games, "Baseline Network", "Challenger Network", show_game=visual, screen=screen)
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

def print_parameters(param_space, values, score):
    # Print the parameters and their values
    param_names = [p.name for p in param_space]
    for param, value in zip(param_names, values):
        print(f"{param}: {value}")
    print(f"Score: {-score}")
    print()  # New line for better readability

if __name__ == "__main__":
    # Run the optimization
    print("Starting optimization...")
    print(f"Parameter space: {param_space}")

    result = gp_minimize(
        func=objective_function,       # Objective function
        dimensions=param_space,        # Parameter space
        n_calls=10,                    # Number of evaluations
        random_state=42                # For reproducibility
    )

    print(f"\nBest Parameters: ")
    print_parameters(param_space, result.x)
    print(f"Best Score: {-result.fun}")  # Negate the score to interpret correctly
    print("------------------------------")
    print(f"All Results: ")
    for i, (x, s) in enumerate(zip(result.x_iters, result.func_vals)):
        print(f"Iteration {i + 1}:")
        print_parameters(param_space, x, s)