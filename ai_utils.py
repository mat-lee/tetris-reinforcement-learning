from const import *
from player import Player

from copy import deepcopy
import random
from typing import Self

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
    
    def make_move(self, turn, move):
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
        if len(other_player.queue.pieces) == 0:
            self.is_done = True

    def get_move_list(self):
        # Using a list of all possible locations the piece can reach
        # Using a queue to determine which boards to check
    
        # Possible locations needs to be larger to acount for blocks with negative x and y
        # O piece        0 to 8
        # Any 3x3 piece -1 to 8
        # I piece       -2 to 8

        player = self.players[self.turn]

        buffer = (2 if player.piece.type == "I" else (0 if player.piece.type == "O" else 1))

        possible_piece_locations = [[[False for o in range (4)] for x in range(COLS + buffer - 1)] for y in range(ROWS)]
        next_location_queue = []
        locations = []

        sim_player = Player()
        sim_player.board = deepcopy(player.board)
        sim_player.piece = deepcopy(player.piece)

        piece = sim_player.piece

        piece.move_to_spawn()
        x = piece.x_0
        y = piece.y_0
        o = piece.rotation

        piece.update_rotation()

        next_location_queue.append((x, y, o))

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
                        and (x, y, o) not in next_location_queue):

                        next_location_queue.append((x, y, o))

            for i in range(1, 4):
                sim_player.try_wallkick(i)

                x = piece.x_0
                y = piece.y_0
                o = piece.rotation

                if possible_piece_locations[y][x + buffer][o] == False and (x, y, o) not in next_location_queue:
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

                for y in reversed(range(ROWS)):
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


class AIPlayer():
    """Class for representing player in NodeState."""

    def __init__(self, board, queue, stats):
        self.board = board
        self.queue = queue
        self.stats = stats

    def place_pieces(self):
        pass    