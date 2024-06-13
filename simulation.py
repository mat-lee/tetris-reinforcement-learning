from ai import *

### File for running the simulation commands

# Create a network if none exist already
DefaultConfig = Config()

if highest_model_ver() == -1:
    create_network(DefaultConfig, save_network=True, plot_model=False)

self_play_loop(DefaultConfig, skip_first_set=False, show_games=True)


# data = load_data(0, 9)

# test_model = load_best_model()
# train_network(test_model, data)

# test_model.save(f"{directory_path}/4_Trained_Model.keras")


### Debugging



# data = load_data(0, 1)[:100]

'''
import time


data = load_data(0, 1)
game=Game()
game.setup()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

for move in data:
    pygame.display.set_caption(str(move[14]))
    pygame.event.get()

    game.players[0].board.grid = move[0]
    game.players[1].board.grid = move[6]

    game.show(screen)
    pygame.display.update()

    time.sleep(0.3)'''













'''
import seaborn as sn
import pandas as pd

scores = {1: {2: '16-24', 4: '21-19', 8: '25-15', 16: '35-5', 32: '23-17', 64: '23-17'}, 2: {1: '24-16', 4: '20-20', 8: '15-25', 16: '30-10', 32: '10-30', 64: '17-23'}, 4: {1: '19-21', 2: '20-20', 8: '21-19', 16: '34-6', 32: '10-30', 64: '22-18'}, 8: {1: '15-25', 2: '25-15', 4: '19-21', 16: '36-4', 32: '22-18', 64: '31-9'}, 16: {1: '5-35', 2: '10-30', 4: '6-34', 8: '4-36', 32: '15-25', 64: '12-28'}, 32: {1: '17-23', 2: '30-10', 4: '30-10', 8: '18-22', 16: '25-15', 64: '29-11'}, 64: {1: '17-23', 2: '23-17', 4: '18-22', 8: '9-31', 16: '28-12', 32: '11-29'}}
for row_dict in scores.values():
    for player in row_dict:
        row_dict[player] = int(row_dict[player].split("-")[0])


df = pd.DataFrame(scores)
print(df.head())

sn.heatmap(df, annot=True)

print("nice")
'''





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