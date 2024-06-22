from ai import *

### File for running the simulation commands

# Create a network if none exist already
DefaultConfig = Config()

if highest_model_ver() == -1:
    create_network(DefaultConfig, save_network=True, plot_model=False)

self_play_loop(DefaultConfig, skip_first_set=False, show_games=True)



### Debugging




'''
data = load_data(0, 10)
print(len(data))

## Grid search battling different parameters


weights = [[1, 0.033], [1, 0.1], [1, 0.33]]

changing_var = weights

configs = [Config(loss_weights=var) for var in changing_var]

networks = [create_network(config, show_summary=False, save_network=False, plot_model=False) for config in configs]

for network in networks:
    train_network(network, data)

interpreters = [get_interpreter(network) for network in networks]

scores={str(title): {} for title in changing_var}

pygame.init()

for i in range(len(changing_var)):
    first_network = interpreters[i]
    first_config = configs[i]
    for j in range(i):
        if i != j:
            second_network = interpreters[j]
            second_config = configs[i] 
            score_1, score_2 = battle_networks_win_loss(first_network, first_config, 
                                                        second_network, second_config,
                                                        network_1_title=changing_var[i], 
                                                        network_2_title=changing_var[j], 
                                                        show_game=True)
            
            scores[str(changing_var[i])][str(changing_var[j])] = f"{score_1}-{score_2}"
            scores[str(changing_var[j])][str(changing_var[i])] = f"{score_2}-{score_1}"


print(scores)
'''