import pygame
import sys
import time

from ai import directory_path, MCTS, load_best_model, get_interpreter, Config
from const import *
from game import Game
from mover import Mover

import cProfile
import pstats

# Load neural network
model = load_best_model()
interpreter = get_interpreter(model)

DefaultConfig = Config()

class Main:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Tetris')
        self.game = Game()
        self.mover = Mover()

    def mainloop(self):
        screen = self.screen
        game = self.game
        mover = self.mover

        game.setup()

        game.players[0].garbage_to_receive = [1 for i in range(18)]
        game.players[1].garbage_to_receive = [1 for i in range(18)]

        while True:
            game.show(screen)

            # Player's move:
            if game.turn == 0 or game.ai_player.game_over == True:
                for event in pygame.event.get():
                    if game.human_player.piece != None:
                        # On key down
                        if event.type == pygame.KEYDOWN:
                            if event.key == k_move_left:
                                game.human_player.move_left()
                                mover.start_left()
                            elif event.key == k_move_right:
                                game.human_player.move_right()
                                mover.start_right()
                            elif event.key == k_soft_drop:
                                game.human_player.move_down()
                                mover.start_down()
                            elif event.key == k_hard_drop:
                                move = (game.human_player.piece.type, 
                                        game.human_player.piece.x_0,
                                        game.human_player.piece.y_0,
                                        game.human_player.piece.rotation)
                                game.make_move(move)
                            elif event.key == k_make_ai_move:
                                if game.players[1].game_over == False:
                                    move, _ = MCTS(DefaultConfig, game, interpreter)
                                    game.make_move(move=move)
                            elif event.key == k_rotate_cw:
                                game.human_player.try_wallkick(1)
                            elif event.key == k_rotate_180:
                                game.human_player.try_wallkick(2)
                            elif event.key == k_rotate_ccw:
                                game.human_player.try_wallkick(3)
                            elif event.key == k_hold:
                                game.human_player.hold_piece()
                            elif event.key == k_undo:
                                game.undo()
                            elif event.key == k_redo:
                                game.redo()
                            elif event.key == k_restart:
                                game.restart()

                        # On key release
                        elif event.type == pygame.KEYUP:
                            # pain
                            if event.key == k_move_left:
                                mover.stop_left()
                            elif event.key == k_move_right:
                                mover.stop_right()
                            elif event.key == k_soft_drop:
                                mover.stop_down()

                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                # DAS, ARR, and Softdrop
                current_time = time.time()

                # This makes das limited by FPS
                if mover.can_lr_das:
                    if mover.lr_das_start_time != None:
                        if current_time - mover.lr_das_start_time > mover.lr_das_counter:
                            if mover.lr_das_direction == "L":
                                game.human_player.move_left()
                            elif mover.lr_das_direction == "R":
                                game.human_player.move_right()      
                            mover.lr_das_counter += ARR/1000

                if mover.can_sd_das:
                    if mover.sd_start_time != None:
                        if current_time - mover.sd_start_time > mover.sd_counter:
                            game.human_player.move_down()
                            mover.sd_counter += (1 / SDF) / 1000

            elif game.turn == 1:
                if game.players[0].game_over == False:
                    # with cProfile.Profile() as pr:
                    #     move, _ = MCTS(game, interpreter)
                    # stats = pstats.Stats(pr)
                    # stats.sort_stats(pstats.SortKey.TIME)
                    # stats.print_stats(20)

                    move, _ = MCTS(DefaultConfig, game, interpreter)
                    game.make_move(move=move)

            pygame.display.update()

if __name__ == "__main__":
    main = Main()
    main.mainloop()