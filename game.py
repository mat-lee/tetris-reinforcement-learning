from ai import AI
from const import *
from history import History
from player import Human

import cProfile
import pstats

class Game:
    """Contains all players and communicates with them."""
    def __init__(self):
        self.human_player = Human()
        self.ai_player = AI()
        
        self.players = [self.ai_player, self.human_player]

        self.history = History()

    # Game methods
    def setup_game(self):
        self.add_bag_to_all()
        for player in self.players:
            player.create_next_piece()
        self.add_history()

    def add_bag_to_all(self):
        bag = self.human_player.queue.generate_bag()
        for player in self.players:
            player.queue.add_bag(bag)

    def make_move(self):
        player_move = None
        if self.human_player.game_over == False:
            player_move = self.human_player.place_piece()
            self.check_garbage(self.human_player)
            self.human_player.create_next_piece() # Create piece after garbage

            # Don't have the AI move if the player is dead
            if self.ai_player.game_over == False:

                with cProfile.Profile() as pr:
                    self.ai_player.make_move(self.human_player, self.ai_player, player_move=player_move)
                stats = pstats.Stats(pr)
                stats.sort_stats(pstats.SortKey.TIME)
                stats.print_stats()
                
                self.check_garbage(self.ai_player)
                self.ai_player.create_next_piece() # Create piece after garbage
        
        # Add history after placing piece
        if (self.human_player.game_over == False or 
            self.ai_player.game_over == False):
            self.add_history()

    def add_history(self):
        # If placing a piece after undoing, get rid if future history
        if self.history.index != None:
            while self.history.index + 1 < len(self.history.boards) and self.history.index >= 0:
                self.history.boards.pop(-1)
        
        # Add on history
        player_boards = []
        for player in self.players:
            if player.piece != None:
                piece = player.piece.copy()
                piece.move_to_spawn()
            else: piece = None
            dictionary = {
                'board': player.board.copy(), 
                'queue': player.queue.copy(), 
                'stats': player.stats.copy(), 
                'piece': piece,
                'held_piece': player.held_piece,
                'garbage_to_receive': player.garbage_to_receive[:],
                'game_over': player.game_over
                }
            player_boards.append(dictionary)
        self.history.boards.append(player_boards)

        # Move history index
        if self.history.index == None:
            self.history.index = 0
        else: self.history.index += 1

    def update_state(self, state):
        for player_idx in range(len(self.players)):
            player_state = state[player_idx]
            for key in player_state:
                setattr(self.players[player_idx], key, player_state[key])

    def undo(self):
        if len(self.history.boards) > 1 and self.history.index != 0:
            state = self.history.boards[self.history.index - 1].copy()
            self.update_state(state)

            self.history.index -= 1
                    
    def redo(self):
        if len(self.history.boards) > self.history.index + 1:
            state = self.history.boards[self.history.index + 1].copy()
            self.update_state(state)
            
            self.history.index += 1
    
    def restart(self):
        for player in self.players:
            player.reset()

        self.add_bag_to_all()
        
        for player in self.players:
            player.create_next_piece()
        
        self.add_history()

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