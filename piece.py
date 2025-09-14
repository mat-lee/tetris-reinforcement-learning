from const import *
from piece_location import PieceLocation

import numpy as np

class Piece:
    '''
    The piece matrix and the center of the piece have a location attribute
    The coordinates are saved and the piece is drawn on top
    '''

    def __init__(self, x_0=0, y_0=0, type=None):
        # x_0 and y_0 are the coordinates of the top left corner of the piece matrix
        self.location = PieceLocation(x=x_0, y=y_0)

        self.type = type
        self.coordinates = np.zeros((4, 2), dtype=np.int8)

    def get_spawn_location(self):
        # Move a newly created "O" piece to the right
        displacement = (1 if self.type == "O" else 0)

        return PieceLocation(3 + displacement, ROWS - SPAWN_ROW)

    def move_to_spawn(self):
        spawn_location = self.get_spawn_location()
        self.location.x = spawn_location.x
        self.location.y = spawn_location.y
        self.coordinates = self.get_self_coords

    def move(self, x_offset=0, y_offset=0):
        self.location.x += x_offset
        self.location.y += y_offset

        # Vectorized addition
        offset_vector = np.array([x_offset, y_offset], dtype=np.int16)
        self.coordinates += offset_vector

        self.location.rotation_just_occurred = False
        self.location.rotation_just_occurred_and_used_last_tspin_kick = False

    def copy(self):
        new_piece = Piece(self.location.x, self.location.y, self.type)
        new_piece.location.rotation = self.location.rotation
        new_piece.location.rotation_just_occurred = self.location.rotation_just_occurred
        new_piece.location.rotation_just_occurred_and_used_last_tspin_kick = self.location.rotation_just_occurred_and_used_last_tspin_kick
        
        new_piece.coordinates = self.coordinates.copy()
    
        return new_piece

    @property
    def get_self_coords(self):
        return Piece.get_mino_coords(self.location.x, self.location.y, self.location.rotation, self.type)
    
    @staticmethod
    def get_mino_coords(x_0, y_0, rotation, type):
        coordinate_list = mino_coords_dict[type][rotation]
        # [col, row]
        # Vectorized addition
        offset_vector = np.array([x_0, y_0], dtype=np.int16)
        
        return coordinate_list + offset_vector