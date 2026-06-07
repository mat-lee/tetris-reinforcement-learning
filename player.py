from board import Board
from piece_queue import Queue
from stats import Stats
from piece import Piece
from const import *

import random
import numpy as np

class Player:
    """Parent class for both human and AI players."""
    def __init__(self, ruleset) -> None:
        self.board = Board()
        self.queue = Queue()
        self.stats = Stats(ruleset)

        self.game_over = False
        self.garbage_to_receive = [] # index 0 spawns first
        self.garbage_to_send = [] # sends and receive are in same order

        self.color = None # Used for turn order

        self.piece = None
        self.held_piece = None

        self.ruleset = ruleset # Specifies tetr.io s1 or s2 spins

    # Game methods
    def create_next_piece(self):
        if len(self.queue.pieces) > 0:
            next_piece = self.queue.pieces.pop(0)
            self.create_piece(next_piece)
        elif self.held_piece != None:
            next_piece, self.held_piece = self.held_piece, None
            self.create_piece(next_piece)

    def create_piece(self, piece_type):
        piece = Piece(piece_dict[piece_type], type=piece_type)
        piece.move_to_spawn()
        if not self.collision(piece.coordinates):
            self.piece = piece
        else:
            # If the block can't spawn lose
            self.game_over = True

    def collision(self, coords):
        return not self.board.is_valid_position(coords)

    @property
    def ghost_y(self):
        piece = self.piece
        base_coords = piece.get_mino_coords(piece.location.x, piece.location.y,
                                            piece.location.rotation, piece.type)
        delta = 1
        while True:
            shifted = [[col, row + delta] for col, row in base_coords]
            if not self.board.is_valid_position(shifted):
                return piece.location.y + delta - 1
            delta += 1

    def can_move(self, piece, x_offset=0, y_offset=0):
        shifted = [[col + x_offset, row + y_offset] for col, row in piece.coordinates]
        return self.board.is_valid_position(shifted)

    def move_right(self):
        if self.can_move(self.piece, x_offset=1):
            self.piece.move(x_offset=1)

    def move_left(self):
        if self.can_move(self.piece, x_offset=-1):
            self.piece.move(x_offset=-1)

    def move_down(self):
        if self.can_move(self.piece, y_offset=1):
            self.piece.move(y_offset=1)

    def try_wallkick(self, dir) -> None:
        """Try to rotate the piece with wallkicks."""
        piece = self.piece
        initial_rotation = piece.location.rotation
        final_rotation = (initial_rotation + dir) % 4

        if piece.type == "I":
            kicktable = i_wallkicks[initial_rotation][final_rotation]
        else:
            kicktable = wallkicks[initial_rotation][final_rotation]

        rotated_coords = piece.get_mino_coords(piece.location.x, piece.location.y, final_rotation, piece.type)
        for kick in kicktable:
            kicked_coords = [[col + kick[0], row - kick[1]] for col, row in rotated_coords]
            if self.board.is_valid_position(kicked_coords):
                piece.location.x += kick[0]
                piece.location.y += -kick[1]
                piece.location.rotation = final_rotation
                piece.coordinates = kicked_coords

                if piece.type == "T":
                    piece.location.rotation_just_occurred = True
                    piece.location.rotation_just_occurred_and_used_last_tspin_kick = (
                        kick == kicktable[-1] and dir != 2
                    )
                return True

    def force_place_piece(self, piece_location):
        self.piece.location = piece_location
        self.piece.coordinates = self.piece.get_self_coords
        return self.place_piece()

    def place_piece(self) -> tuple:
        piece = self.piece
        grid = self.board.grid
        stats = self.stats

        place_y = self.ghost_y
        # If the piece was placed in the air, it does not count as a spin
        if piece.location.y != place_y:
            piece.location.rotation_just_occurred = False
            piece.location.rotation_just_occurred_and_used_last_tspin_kick = False

        rows = []
        cleared_rows = []

        is_tspin = False
        is_mini = False
        is_all_clear = False

        # Check for a t-spin
        if piece.type == "T" and piece.location.rotation_just_occurred:
            corners = [[0, 0], [2, 0], [2, 2], [0, 2]]
            corner_filled = [
                not self.board.is_valid_position([[corners[i][0] + piece.location.x, corners[i][1] + place_y]])
                for i in range(4)
            ]

            if sum(corner_filled) >= 3:
                is_tspin = True

            # Two corner rule + exception
            if not (corner_filled[piece.location.rotation] and corner_filled[(piece.location.rotation + 1) % 4]) and not piece.location.rotation_just_occurred_and_used_last_tspin_kick:
                is_mini = True

        # Check for s2 all spins
        self.piece.coordinates = self.piece.get_self_coords

        if self.ruleset == 's2' and not is_tspin:
            offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
            for x_offset, y_offset in offsets:
                if self.can_move(piece, x_offset=x_offset, y_offset=y_offset):
                    break
            else:
                is_mini = True

        # Place the pieces and check rows that minos will be placed
        for col, row in piece.get_mino_coords(piece.location.x, place_y, piece.location.rotation, piece.type):
            grid[row][col] = piece.type
            if row not in rows:
                rows.append(row)

        self.piece = None

        # Check which rows should be cleared
        for row in rows:
            if all(mino != 0 for mino in grid[row]):
                cleared_rows.append(row)

        rows_cleared = len(cleared_rows)

        if len(cleared_rows) > 0:
            cleared_rows.sort()

        if cleared_rows:
            self.board.grid = np.vstack([
                np.zeros((len(cleared_rows), COLS), dtype=object),
                np.delete(self.board.grid, cleared_rows, axis=0)
            ])
            grid = self.board.grid

        if rows_cleared > 0:
            is_all_clear = not np.any(self.board.grid[rows_cleared:] != 0)

        attack = stats.get_attack(rows_cleared, is_tspin, is_mini, is_all_clear, piece.type)
        stats.pieces += 1

        if attack > 0:
            column = random.randint(0, 9)
            self.garbage_to_send.extend([column] * attack)

        return rows_cleared

    def hold_piece(self):
        if self.held_piece == None:
            self.held_piece = self.piece.type
            self.piece = None
            self.create_next_piece()
        else:
            temp = self.held_piece
            if self.piece == None:
                self.held_piece = None
            else:
                self.held_piece = self.piece.type
            self.create_piece(temp)

    def spawn_garbage(self):
        self.board.create_garbage(self.garbage_to_receive)
        self.garbage_to_receive = []

    def reset(self):
        self.board = Board()
        self.queue = Queue()
        self.stats = Stats(self.ruleset)
        self.piece = None
        self.held_piece = None
        self.garbage_to_receive = []
        self.game_over = False

    # AI methods
    def copy(self):
        new_player = Player(self.ruleset)

        new_player.board = self.board.copy()
        new_player.queue = self.queue.copy()
        new_player.stats.pieces = self.stats.pieces
        new_player.stats.b2b = self.stats.b2b
        new_player.stats.b2b_level = self.stats.b2b_level
        new_player.stats.combo = self.stats.combo
        new_player.game_over = self.game_over
        if self.piece != None:
            new_player.piece = self.piece.copy()
        new_player.held_piece = self.held_piece
        new_player.garbage_to_receive = self.garbage_to_receive[:]
        new_player.color = self.color

        return new_player

    def copy_no_board(self):
        """Create a player copy that shares the board reference.
        Safe only when the caller guarantees the board will not be mutated."""
        new_player = object.__new__(Player)
        new_player.board = self.board  # shared reference — no copy
        new_player.queue = self.queue.copy()
        new_player.stats = self.stats.copy()
        new_player.game_over = self.game_over
        new_player.garbage_to_receive = self.garbage_to_receive[:]
        new_player.garbage_to_send = []
        new_player.color = self.color
        new_player.piece = self.piece.copy() if self.piece is not None else None
        new_player.held_piece = self.held_piece
        new_player.ruleset = self.ruleset

        return new_player


class Human(Player):
    def __init__(self, ruleset) -> None:
        super().__init__(ruleset)
        self.color = 0

class AI(Player):
    def __init__(self, ruleset) -> None:
        super().__init__(ruleset)
        self.color = 1
