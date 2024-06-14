from const import *
from game import Game
from ai import Config, get_interpreter, load_best_model, MCTS

import pygame

import cProfile
import pstats

import time

# Test speed of a game
def time_move_matrix():
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

time_move_matrix()

# 100 iter: 
# Initial:                        0.370
# Deque:                          0.381
# Using pop immediately:          0.362
# Using minor sd instead of full: 0.364
# Keeping piece coordinates:      0.812