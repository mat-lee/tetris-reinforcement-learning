from const import *
from history import History
from player import Human, AI

class Game:
    """Contains all players and communicates with them."""
    def __init__(self):
        self.human_player = Human()
        self.ai_player = AI()
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

    def make_move(self, move, add_bag=True, add_history=True, send_garbage=True):
        player = self.players[self.turn]

        # Move: (Piece type, col, row, rotation) or (Policy index, col, row)
        if player.game_over == False:

            piece = None
            row = None
            col = None
            rotation = None
            # If the move is straight from the policy (policy index, col, row),
            # convert it to piece and rotation
            if len(move) == 3:
                piece, rotation = policy_index_to_piece[move[0]]
                col = move[1]
                row = move[2]
            elif len(move) == 4:
                piece, col, row, rotation = move

            # If the player doesn't have an active piece, the ai wants it to hold
            if player.piece == None:
                player.hold_piece()

            # If the piece it wants to place is not the same as the active piece, hold
            elif piece != player.piece.type:
                player.hold_piece()

            player.force_place_piece(col, row, rotation)

            lines = self.check_garbage(send_garbage)

            player.create_next_piece()

            if (add_bag == True and len(player.queue.pieces) < 5): # In MCTS don't add pieces to the queue
                self.add_bag_to_all()

            # Add history after placing piece
            if (add_history == True and self.human_player.game_over == False and self.turn == 1): # IN MCTS don't add history
                self.add_history()

            self.turn = 1 - self.turn # 1 to 0

            return lines

    def check_garbage(self, send_garbage):
        active_player = self.players[self.turn]
        other_player = self.players[1 - self.turn]

        # Checks for sending garbage, sends garbage, and canceling
        while len(active_player.garbage_to_send) > 0 and len(active_player.garbage_to_receive) > 0: # Cancel garbage
            # Remove first elements
            active_player.garbage_to_send.pop(0)
            active_player.garbage_to_receive.pop(0)
        
        if len(active_player.garbage_to_send) > 0:
            other_player.garbage_to_receive += active_player.garbage_to_send # Send garbage
            active_player.garbage_to_send = [] # Stop sending garbage
        
        lines = 0
        
        if len(active_player.garbage_to_receive) > 0:
            lines = active_player.spawn_garbage(send_garbage)

        return lines

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
        new_game = Game()
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