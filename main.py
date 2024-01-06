import pygame
import sys
import time

from const import *
from game import Game
from mover import Mover

class Main:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode( (WIDTH, HEIGHT))
        pygame.display.set_caption('Tetris')
        self.game = Game()
        self.mover = Mover()

    def mainloop(self):
        screen = self.screen
        game = self.game
        mover = self.mover

        game.setup_game()

        while True:
            game.show_bg(screen)

            for player in game.players:
                player.show_player(screen)
 
                if len(player.queue.pieces) < 14:
                    game.add_bag_to_all()

            for event in pygame.event.get():
                if game.human_player.piece != None:
                    # On key down
                    if event.type == pygame.KEYDOWN:
                        if event.key == k_move_left:
                            mover.start_left()
                        elif event.key == k_move_right:
                            mover.start_right()
                        elif event.key == k_soft_drop:
                            mover.start_down()
                        elif event.key == k_hard_drop:
                            game.place_piece()
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
                        mover.reset_counter(is_sd=False)
                        mover.reset_counter(is_sd=True)
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

            pygame.display.update()

            # Moving pieces snippet 
            # Add piece moves to a queue
            current_time = time.time()
            mover.update_queue(current_time, is_sd=False)
            mover.update_queue(current_time, is_sd=True)

            while len(mover.movement_string) > 0:
                str = mover.movement_string

                if str[0] == "L":
                    game.human_player.move_left()
                    mover.reset_counter(is_sd=False)
                elif str[0] == "R":
                    game.human_player.move_right()
                    mover.reset_counter(is_sd=False)
                elif str[0] == "D":
                    game.human_player.move_down()
                    mover.reset_counter(is_sd=True)
                mover.movement_string = str[1:]

main = Main()
main.mainloop()