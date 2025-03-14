from const import *
from history import History
from player import Human, AI

class Game:
    """Contains all players and communicates with them."""
    def __init__(self, ruleset):
        self.ruleset = ruleset

        self.human_player = Human(ruleset)
        self.ai_player = AI(ruleset)
        self.turn = 0
        
        self.players = [self.human_player, self.ai_player]

        self.history = History()

    # Game methods
    def setup(self):
        self.add_bag_to_all()
        for player in self.players:
            player.create_next_piece()
        self.add_history()

    def add_bag_to_all(self):
        # Randomize bags for players
        for player in self.players:
            bag = player.queue.generate_bag()
            player.queue.add_bag(bag)

    def move_piece(self, move):
        # This function moves the the current player's piece. 
        # move: (Policy index, was_just_rotated, col, row)
        player = self.players[self.turn]

        policy_index, col, row = move
        piece, rotation = policy_index_to_piece[policy_index]

        # If the player doesn't have an active piece, the ai wants it to hold
        if player.piece == None:
            player.hold_piece()

        # If the piece it wants to place is not the same as the active piece, hold
        elif piece != player.piece.type:
            player.hold_piece()

        player.piece.x_0 = col
        player.piece.y_0 = row
        player.piece.rotation = rotation

    def place(self, add_bag=True, add_history=True):
        # It places the piece, updates garbage, pieces, bag, and history.
        player = self.players[self.turn]

        if player.game_over == False:
            rows_cleared = player.place_piece()

            if rows_cleared > 0:
                is_last_move_line_clear = True
            else:
                is_last_move_line_clear = False

            self.check_garbage(is_last_move_line_clear)

            player.create_next_piece()

            if (add_bag == True and len(player.queue.pieces) < 5): # In MCTS don't add pieces to the queue
                self.add_bag_to_all()

            # Add history after placing piece
            if (add_history == True and self.turn == 1): # IN MCTS don't add history
                self.add_history()

            self.turn = 1 - self.turn # 1 to 0

    def make_move(self, move, add_bag=True, add_history=True):
        self.move_piece(move)
        self.place(add_bag=add_bag, add_history=add_history)

    def check_garbage(self, is_last_move_line_clear):
        active_player = self.players[self.turn]
        other_player = self.players[1 - self.turn]

        # Checks for sending garbage, sends garbage, and canceling
        # If cancelling, don't receive garbage
        while len(active_player.garbage_to_send) > 0 and len(active_player.garbage_to_receive) > 0: # Cancel garbage
            # Remove first elements
            active_player.garbage_to_send.pop(0)
            active_player.garbage_to_receive.pop(0)

        # If the active player didn't make a line clear, give it the garbage
        if len(active_player.garbage_to_receive) > 0 and not is_last_move_line_clear:
            active_player.spawn_garbage()

        if len(active_player.garbage_to_send) > 0:
            other_player.garbage_to_receive += active_player.garbage_to_send # Send garbage
            active_player.garbage_to_send = [] # Stop sending garbage


    def add_history(self):
        # If placing a piece after undoing, get rid of future history
        if self.history.index != None:
            while self.history.index + 1 < len(self.history.states) and self.history.index >= 0:
                self.history.states.pop(-1)
        
        # Add on history
        player_states = []
        for player in self.players:
            if player.piece != None:
                piece = player.piece.copy()
                piece.move_to_spawn()
            else: piece = None
            # Only need to copy certain variables
            # Don't need to repeat history
            dictionary = {
                'board': player.board.copy(), 
                'queue': player.queue.copy(), 
                'stats': player.stats.copy(), 
                'game_over': player.game_over,
                'piece': piece,
                'held_piece': player.held_piece,
                'garbage_to_receive': player.garbage_to_receive[:]
                }
            player_states.append(dictionary)
        self.history.states.append(player_states)

        # Move history index
        if self.history.index == None:
            self.history.index = 0
        else: self.history.index += 1

    def update_state(self, state):
        # Changes the state of both players to a given input state
        for player_idx in range(len(self.players)):
            player_state = state[player_idx] # Make sure to copy player info

            if player_state["piece"] != None:
                piece = player_state["piece"].copy()
                piece.move_to_spawn()
            else: piece = None

            copied_dict = {
                'board': player_state["board"].copy(), 
                'queue': player_state["queue"].copy(), 
                'stats': player_state["stats"].copy(), 
                'game_over': player_state["game_over"],
                'piece': piece,
                'held_piece': player_state["held_piece"],
                'garbage_to_receive': player_state["garbage_to_receive"][:]
            }

            for key in copied_dict:
                setattr(self.players[player_idx], key, copied_dict[key])

    def undo(self):
        if len(self.history.states) > 1 and self.history.index != 0:
            state = self.history.states[self.history.index - 1]
            self.update_state(state)

            self.history.index -= 1
                        
    def redo(self):
        if len(self.history.states) > self.history.index + 1:
            state = self.history.states[self.history.index + 1]
            self.update_state(state)
            
            self.history.index += 1
    
    def restart(self):
        for player in self.players:
            player.reset()

        self.add_bag_to_all()
        
        for player in self.players:
            player.create_next_piece()
        
        self.add_history()

        self.players[1].hold_piece()

        self.turn = 0
    
    @property
    def is_terminal(self):
        for player in self.players:
            if player.game_over == True:
                return True
        return False
    
    @property
    def no_move(self):
        if self.players[self.turn].piece == None and self.players[self.turn].held_piece == None:
            return True
        return False
    
    @property
    def winner(self):
        status = [player.game_over for player in self.players]
        if status[0] == True:
            return 1
        elif status[1] == True:
            return 0
        else:
            return -1

    def copy(self):
        new_game = Game(self.ruleset)
        new_game.players = [player.copy() for player in self.players]
        new_game.turn = self.turn

        return new_game

    # Show methods
    def show(self, surface):
        self.show_bg(surface)
        for player in self.players:
            player.show(surface)

    def show_bg(self, surface):
        surface.fill((0, 0, 0))