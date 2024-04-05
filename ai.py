from const import *
from player import Player

import time

import math

class AI(Player):
    def __init__(self) -> None:
        super().__init__()
        self.draw_coords = (WIDTH/2, 0)
    
    def create_move_list(self, board, _piece):
        TIME_1 = time.time()
        # Using a list of all possible locations the piece can reach
        # Using a queue to determine which boards to check
    
        # Possible locations needs to be larger to acount for blocks with negative x and y
        # O piece        0 to 8
        # Any 3x3 piece -1 to 8
        # I piece       -2 to 8

        buffer = (2 if _piece.type == "I" else (0 if _piece.type == "O" else 1))

        possible_piece_locations = [[[False for x in range (4)] for x in range(COLS + buffer - 1)] for y in range(ROWS)]
        next_location_queue = []
        locations = []

        demo_player = Player()
        demo_player.board = board
        demo_player.piece = _piece

        piece = demo_player.piece

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

                if demo_player.can_move(x_offset=move[0], y_offset=move[1]): # Check this first to avoid index errors
                    if (possible_piece_locations[y][x + buffer][o] == False 
                        and (x, y, o) not in next_location_queue):

                        next_location_queue.append((x, y, o))

            for i in range(1, 4):
                demo_player.try_wallkick(i)

                x = piece.x_0
                y = piece.y_0
                o = piece.rotation

                if possible_piece_locations[y][x + buffer][o] == False and (x, y, o) not in next_location_queue:
                    next_location_queue.append((x, y, o))

                piece.x_0, piece.y_0, piece.rotation = next_location_queue[0]

                piece.update_rotation()

            next_location_queue.pop(0)

        # Remove entries that can move downwards
        for o in range(4): # Smalleswet number of operations
            demo_player.piece.rotation = o
            demo_player.piece.update_rotation()

            for x in range(COLS + buffer - 1):
                demo_player.piece.x_0 = x - buffer

                for y in reversed(range(ROWS)):
                    if possible_piece_locations[y][x][o] == True:
                        demo_player.piece.y_0 = y
                        if not demo_player.can_move(y_offset=1):
                            locations.append((x - buffer, y, o))

        return locations, time.time()-TIME_1

    def make_move(self):
        self.place_piece()


if __name__ == "__main__":

    from board import Board
    from piece import Piece

    test = AI()

    piece_type = 'T'
    print(test.create_move_list(Board(), Piece(piece_dict[piece_type], type=piece_type)))