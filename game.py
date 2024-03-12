from const import *
from player import Human, AI

import copy
import random

minos = "ZLOSIJT"

class Game:
    """Contains all players and communicates with them."""
    def __init__(self):
        self.human_player = Human()
        self.ai_player = AI()
        
        self.players = [self.ai_player, self.human_player]

        self.history = []

    # Game methods
    def generate_bag(self):
        mino_list = list(minos)
        random.shuffle(mino_list)
        return mino_list

    def setup_game(self):
        self.add_bag_to_all()
        for player in self.players:
            player.create_next_piece()

    def add_bag_to_all(self):
        bag = self.generate_bag()
        for player in self.players:
            player.queue.add_bag(bag)

    def make_move(self):
        if (self.human_player.game_over == False or 
            self.ai_player.game_over == False):
            self.add_history()
        if self.human_player.game_over != True:
            self.human_player.place_piece()
            self.check_garbage(self.human_player)
        if self.ai_player.game_over != True:
            self.ai_player.make_move()
            self.check_garbage(self.ai_player)
        
    def add_history(self):
        player_history = []
        for player in self.players:
            piece = copy.deepcopy(player.piece)
            if piece != None:
                piece.move_to_spawn()
            dictionary = {
                'board': copy.deepcopy(player.board), 
                'queue': copy.deepcopy(player.queue), 
                'stats': copy.deepcopy(player.stats), 
                'piece': piece,
                'held_piece': copy.deepcopy(player.held_piece),
                'game_over': copy.deepcopy(player.game_over)
                }
            player_history.append(dictionary.copy())
        self.history.append(player_history)

    def undo(self):
        if len(self.history) > 0:
            state = self.history[-1]
            for i in range(len(self.players)):
                player_state = state[i]
                for key in player_state:
                    setattr(self.players[i], key, player_state[key])
                
            self.history.pop(-1)
    
    def check_garbage(self, player):
        other_player = [x for x in self.players if x != player][0]

        # Checks for sending garbage, sends garbage, and canceling
        while len(player.garbage_to_send) > 0 and len(player.garbage_to_receive) > 0: # Cancel garbage
            # Remove first elements
            player.garbage_to_send.pop(0)
            player.garbage_to_receive.pop(0)
        
        if len(player.garbage_to_send) > 0:
            other_player.garbage_to_receive += player.garbage_to_send # Send garbage
            player.garbage_to_send = [] # Stop sending garbage
        
        if len(player.garbage_to_receive) > 0:
            player.spawn_garbage()
    
    # Show methods
    def show_bg(self, surface):
        surface.fill((0, 0, 0))

        '''grey = (100, 100, 100)
        width = 4

        # Main Box
        pygame.draw.line(surface, grey, 
                                   (E_BUFFER + HOLD_WIDTH, N_BUFFER), 
                                   (E_BUFFER + HOLD_WIDTH + COLS * MINO_SIZE, N_BUFFER), width=width)
        pygame.draw.line(surface, grey, 
                                   (E_BUFFER + HOLD_WIDTH + COLS * MINO_SIZE, N_BUFFER), 
                                   (E_BUFFER + HOLD_WIDTH + COLS * MINO_SIZE, N_BUFFER + ROWS * MINO_SIZE), width=width)
        pygame.draw.line(surface, grey, 
                                   (E_BUFFER + HOLD_WIDTH + COLS * MINO_SIZE, N_BUFFER + ROWS * MINO_SIZE), 
                                   (E_BUFFER + HOLD_WIDTH, N_BUFFER + ROWS * MINO_SIZE), width=width)
        pygame.draw.line(surface, grey, 
                                   (E_BUFFER + HOLD_WIDTH, N_BUFFER + ROWS * MINO_SIZE),
                                   (E_BUFFER + HOLD_WIDTH, N_BUFFER), width=width)'''