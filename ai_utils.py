from const import *
from piece_queue import Queue
from player import Player

import random

# Dedicated file for functions for using for AI 
# More specialized than the original functions

class NodeState():
    """Node class for storing the game in the tree.
    
    Similar to the game class, but is more specialized for AI functions and has different variables"""

    def __init__(self, players=None, turn=None, move=None) -> None:
        self.players = players
        self.turn = turn
        self.move = move # The move which resulted in this state

        # Search tree variables
        # Would be stored in each edge, but this is essentially the same thing
        self.N = 1
        self.W = 0
        self.Q = 0
        self.P = 0

        self.value = None
        self.policy = None
        self.move_list = None

        self.is_done = False
    
    def check_if_done(self):
        for player in self.players:
            if player.game_over == True:
                self.is_done = True
    
    def make_move(self, turn, move): # e^-5
        player = self.players[turn]
        other_player = self.players[turn - 1]

        player.force_place_piece(*move)

        # Checks for sending garbage, sends garbage, and canceling
        while len(player.garbage_to_send) > 0 and len(player.garbage_to_receive) > 0: # Cancel garbage
            # Remove first elements
            player.garbage_to_send.pop(0)
            player.garbage_to_receive.pop(0)

        if len(player.garbage_to_send) > 0:
            other_player.garbage_to_receive += player.garbage_to_send # Send garbage
            player.garbage_to_send = [] # Stop sending garbage
        
        elif len(player.garbage_to_receive) > 0:
            player.spawn_garbage()
        
        player.create_next_piece()

        # Check if the game is over:
        self.check_if_done()

        # Stop searching if the other player doesn't have a piece
        if len(other_player.queue.pieces) == 0 and other_player.piece == None:
            self.is_done = True

    def get_move_list(self): # e^-3
        # Using a list of all possible locations the piece can reach
        # Using a queue to determine which boards to check
    
        # Possible locations needs to be larger to acount for blocks with negative x and y
        # O piece        0 to 8
        # Any 3x3 piece -1 to 8
        # I piece       -2 to 8

        def get_highest_row(grid):
            for row in range(len(grid)):
                for col in range(len(grid[0])):
                    if grid[row][col] != 0:
                        highest_row = row
                        return highest_row
            
            # Else return the floor
            return len(grid)

        # Load the state
        player = self.players[self.turn]
        sim_player = Player()
        sim_player.board = player.board.copy()
        sim_player.piece = player.piece.copy()

        piece = sim_player.piece

        # On the left hand side, blocks can have negative x
        buffer = (2 if piece.type == "I" else (0 if piece.type == "O" else 1))

        # No piece can be placed in the bottom row; ROWS - 1
        possible_piece_locations = [[[False for o in range (4)] for x in range(COLS + buffer - 1)] for y in range(ROWS - 1)]
        next_location_queue = []
        locations = []

        # Start the piece at the highest point it can be placed
        highest_row = get_highest_row(sim_player.board.grid)
        starting_row = max(highest_row - len(piece.matrix), 0)
        piece.y_0 = starting_row

        x = piece.x_0
        y = piece.y_0
        o = piece.rotation

        piece.update_rotation()

        next_location_queue.append((x, y, o))

        # Search through the queue
        while len(next_location_queue) > 0:
            piece.x_0, piece.y_0, piece.rotation = next_location_queue[0]

            piece.update_rotation()

            possible_piece_locations[piece.y_0][piece.x_0 + buffer][piece.rotation] = True

            for move in [[1, 0], [-1, 0], [0, 1]]:
                x = piece.x_0 + move[0]
                y = piece.y_0 + move[1]
                o = piece.rotation

                if sim_player.can_move(x_offset=move[0], y_offset=move[1]): # Check this first to avoid index errors
                    if (possible_piece_locations[y][x + buffer][o] == False 
                        and (x, y, o) not in next_location_queue
                        and y >= 0): # Avoid negative indexing

                        next_location_queue.append((x, y, o))


            for i in range(1, 4):
                sim_player.try_wallkick(i)

                x = piece.x_0
                y = piece.y_0
                o = piece.rotation

                if (possible_piece_locations[y][x + buffer][o] == False
                    and (x, y, o) not in next_location_queue
                    and y >= 0): # Avoid negative indexing
                    next_location_queue.append((x, y, o))

                piece.x_0, piece.y_0, piece.rotation = next_location_queue[0]

                piece.update_rotation()

            next_location_queue.pop(0)

        # Remove entries that can move downwards
        for o in range(4): # Smallest number of operations
            sim_player.piece.rotation = o
            sim_player.piece.update_rotation()

            for x in range(COLS + buffer - 1):
                sim_player.piece.x_0 = x - buffer

                for y in reversed(range(ROWS - 1)):
                    if possible_piece_locations[y][x][o] == True:
                        sim_player.piece.y_0 = y
                        if not sim_player.can_move(y_offset=1):
                            locations.append((x - buffer, y, o))

        return locations 

    def get_value(self):
        # Use a neural net to update self value
        return random.random()
    
    def get_policy(self):
        return None

    """Class for representing all necessary player info.
    
    Deepcopying player states was extremely slow, 
    so this class is necessary to copy info 
    through the tree"""

'''class PlayerState():
    def __init__(self, 
                 grid: list, 
                 queue: list, 
                 pieces: int, 
                 b2b: int, 
                 b2b_level: int,
                 combo: int,
                 game_over: bool,
                 piece: str,
                 held_piece: str):
        
        self.grid = grid
        self.queue = queue
        self.pieces = pieces
        self.b2b = b2b
        self.b2b_level = b2b_level
        self.combo = combo
        self.game_over = game_over
        self.piece = piece
        self.held_piece = held_piece

    def return_copy(self):
        return PlayerState([x[:] for x in self.grid], 
                           copy.copy(self.queue),
                           self.pieces,
                           self.b2b,
                           self.b2b_level,
                           self.combo,
                           self.game_over,
                           self.piece,
                           self.held_piece)

    def load_to_player(self, player):
        player.board.grid = [x[:] for x in self.grid]
        player.queue = Queue(pieces=self.queue)
        player.stats.pieces = self.pieces
        player.stats.b2b = self.b2b
        player.stats.b2b_level = self.b2b_level
        player.stats.combo = self.combo
        player.game_over = self.game_over
        player.piece = self.piece
        player.create_piece(self.piece)
        player.held_piece = self.held_piece'''


# Many copies of board states is unavoidable
# So, find what needs to be copied and represent in the simplest form
# Use a dict; dict.copy() will be much faster than deepcopy
# And don't have classes in the dictionary
# For now, the neural nets will just use both board states

# Also find a way to copy the NodeState
    # Nevermind

# Using deepcopy:                  100 iter in 36.911 s
# Using copy functions in classes: 100 iter in 1.658 s