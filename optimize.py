from ai import *
from tests import battle_networks_win_loss

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

param_space = [
    # Real(0.0, 0.9, name='dropout'),
    Real(0.0, 1.0, name='policy_loss_weight'),
    Real(0.0, 1.0, name='DIRICHLET_ALPHA'),
    # Categorical([False, True], name='use_playout_cap_randomization'),
    # Categorical(['reduction', 'absolute'], name='FpuStrategy'),
    # Real(0.0, 1.0, name='FpuValue'),
    # Categorical([False, True], name='use_forced_playouts_and_policy_target_pruning'),
    # Integer(0.0, 3.0, name='CForcedPlayout'),
    Real(1, 100, name='CPUCT'),
]

training_games = 10
training_loops = 4
eval_games = 40
visual = True
screen = pygame.display.set_mode((WIDTH, HEIGHT))
i = 0

def evaluate_challenger(baseline_config, challenger_config, challenger_interpreter, num_games, visual, screen):
    baseline_interpreter = get_interpreter(load_best_model(baseline_config))

    wins = battle_networks_win_loss(baseline_interpreter, baseline_config, challenger_interpreter, challenger_config, num_games, "Baseline Network", "Challenger Network", show_game=visual, screen=screen)
    chal_wins = wins[1]
    eval = chal_wins / num_games # Win more -> higher eval
    return eval

@use_named_args(param_space)
def objective_function(**params):
    global i
    i += 1
    print(i, params)
    baseline_config = Config()
    challenger_config = Config()
    for param in params:
        if param == 'policy_loss_weight':
            challenger_config.loss_weights = [1, params[param]]
        else:
            setattr(challenger_config, param, params[param])
    
    challenger_network = instantiate_network(challenger_config, show_summary=False, save_network=False, plot_model=False)
    
    for _ in range(training_loops):
        interpreter = get_interpreter(challenger_network)
        set = make_training_set(challenger_config, interpreter, training_games, save_game=False, show_game=visual, screen=screen)
        
        train_network(challenger_config, challenger_network, set)

        del set
        gc.collect()
    
    challenger_interpreter = get_interpreter(challenger_network)

    result = evaluate_challenger(baseline_config, challenger_config, challenger_interpreter, eval_games, visual, screen)
    
    # Negate score because skopt minimizes the objective
    return -result

result = gp_minimize(
    func=objective_function,       # Objective function
    dimensions=param_space,        # Parameter space
    n_calls=20,                    # Number of evaluations
    random_state=42                # For reproducibility
)

print(f"Best Parameters: {result.x}")
print(f"Best Score: {-result.fun}")  # Negate the score to interpret correctly
print([print(x) for x in result.x_iters])