import pygame
import sys
import time

from ai import directory_path, MCTS, get_interpreter, load_best_model, Config
from const import *
from game import Game
from mover import Mover

import cProfile
import pstats

import random

DefaultConfig = Config(training=False,MAX_ITER=160, model='keras')

# Load neural network
model = load_best_model(DefaultConfig)
interpreter = get_interpreter(model)

class Main:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Tetris')
        self.game = Game(DefaultConfig.ruleset)
        self.mover = Mover()

        self.human_turn = 1
        self.ai_turn = 0
        self.human_player = self.game.players[self.human_turn]
        self.ai_player = self.game.players[self.ai_turn]

        assert(self.human_turn + self.ai_turn == 1)

    def mainloop(self):
        screen = self.screen
        game = self.game
        mover = self.mover

        game.setup()

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

                # Only for first player
                if game.turn == self.human_turn and not game.is_terminal:
                    if self.human_player.piece != None:
                        # On key down
                        if event.type == pygame.KEYDOWN:
                            if event.key == k_move_left:
                                self.human_player.move_left()
                                mover.start_left()
                            elif event.key == k_move_right:
                                self.human_player.move_right()
                                mover.start_right()
                            elif event.key == k_soft_drop:
                                self.human_player.move_down()
                                mover.start_down()
                            elif event.key == k_hard_drop:
                                game.place()
                            elif event.key == k_rotate_cw:
                                self.human_player.try_wallkick(1)
                            elif event.key == k_rotate_180:
                                self.human_player.try_wallkick(2)
                            elif event.key == k_rotate_ccw:
                                self.human_player.try_wallkick(3)
                            elif event.key == k_hold:
                                self.human_player.hold_piece()

                            # Helper keybinds
                            elif event.key == k_make_ai_move:
                                if self.ai_player.game_over == False:
                                    move, _, _ = MCTS(DefaultConfig, game, interpreter)
                                    game.place()
                            elif event.key == k_add_garbage:
                                for player in game.players:
                                    player.garbage_to_receive.append(random.randint(0, 9))
                            elif event.key == k_switch:
                                game.players[0].board, game.players[1].board = game.players[1].board, game.players[0].board
                            elif event.key == k_print_board:
                                print(game.players[game.turn].board.grid)

                        # On key release
                        elif event.type == pygame.KEYUP:
                            # pain
                            if event.key == k_move_left:
                                mover.stop_left()
                            elif event.key == k_move_right:
                                mover.stop_right()
                            elif event.key == k_soft_drop:
                                mover.stop_down()

            if game.turn == self.human_turn and not game.is_terminal:
                # DAS, ARR, and Softdrop
                current_time = time.time()

                # This makes das limited by FPS
                if mover.can_lr_das:
                    if mover.lr_das_start_time != None:
                        if current_time - mover.lr_das_start_time > mover.lr_das_counter:
                            if mover.lr_das_direction == "L":
                                self.human_player.move_left()
                            elif mover.lr_das_direction == "R":
                                self.human_player.move_right()      
                            mover.lr_das_counter += ARR/1000

            if mover.can_sd_das:
                if mover.sd_start_time != None:
                    if current_time - mover.sd_start_time > mover.sd_counter:
                        self.human_player.move_down()
                        mover.sd_counter += (1 / SDF) / 1000

            # AI's turn
            if game.turn == self.ai_turn and not game.is_terminal:
                if game.players[0].game_over == False:
                    # with cProfile.Profile() as pr:
                    #     move, _ = MCTS(game, interpreter)
                    # stats = pstats.Stats(pr)
                    # stats.sort_stats(pstats.SortKey.TIME)
                    # stats.print_stats(20)

                    move, _, _= MCTS(DefaultConfig, game, interpreter)
                    game.make_move(move=move)

            pygame.display.update()

if __name__ == "__main__":
    main = Main()
    main.mainloop()