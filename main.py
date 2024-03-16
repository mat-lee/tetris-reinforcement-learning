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
                            game.human_player.move_left()
                            mover.start_left()
                        elif event.key == k_move_right:
                            game.human_player.move_right()
                            mover.start_right()
                        elif event.key == k_soft_drop:
                            game.human_player.move_down()
                            mover.start_down()
                        elif event.key == k_hard_drop:
                            game.make_move()
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

            # DAS, ARR, and Softdrop
            current_time = time.time()

            ### This code, while cool, does not matter
            ### I still need to consider sliding off and corners
            '''
            # Stop code from running if das is into a wall  
            if ((mover.lr_das_direction == "L" and not game.human_player.can_move_left())
                or (mover.lr_das_direction == "R" and not game.human_player.can_move_right())):
                mover.can_lr_das = False
            
            # Stop code from runing if sd is into floor
            if (mover.sd_held == True and not game.human_player.can_move_down()):
                mover.can_sd_das = False
            '''

            if mover.can_lr_das:
                if mover.lr_das_start_time != None:
                    if current_time - mover.lr_das_start_time > mover.lr_das_counter:
                        if mover.lr_das_direction == "L":
                            game.human_player.move_left()
                        elif mover.lr_das_direction == "R":
                            game.human_player.move_right()      
                        mover.lr_das_counter += ARR/1000
#                        print(mover.can_lr_das,mover.lr_das_direction, game.human_player.can_move_left(), current_time)

            if mover.can_sd_das:
                if mover.sd_start_time != None:
                    if current_time - mover.sd_start_time > mover.sd_counter:
                        game.human_player.move_down()
                        mover.sd_counter += (1 / SDF) / 1000
#                        print(mover.can_sd_das, game.human_player.can_move_down(),current_time)

main = Main()
main.mainloop()