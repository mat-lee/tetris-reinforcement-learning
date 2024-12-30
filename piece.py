from const import *

class Piece:
    '''
    The piece matrix and the center of the piece have a location attribute
    The coordinates are saved and the piece is drawn on top
    '''

    def __init__(self, x_0=0, y_0=0, type=None):
        # Coords describe the row of the top left value in the matrix
        self.x_0 = x_0
        self.y_0 = y_0
        self.type = type
        self.rotation = 0 # 1 is right, 2 is 180, 3 is left
        self.coordinates = [[], [], [], []]
        self.was_just_rotated = False # If the last input on the piece was a rotation

    def get_spawn_location(self):
        # Move a newly created "O" piece to the right
        displacement = (1 if self.type == "O" else 0)

        return (3 + displacement, ROWS - SPAWN_ROW)

    def move_to_spawn(self):
        self.x_0, self.y_0 = self.get_spawn_location()
        self.coordinates = self.get_self_coords

    def move(self, x_offset=0, y_offset=0):
        self.x_0 += x_offset
        self.y_0 += y_offset
        self.coordinates = [[col + x_offset, row + y_offset] for col, row in self.coordinates]

    def copy(self):
        new_piece = Piece(piece_dict[self.type], type=self.type)
        new_piece.x_0 = self.x_0
        new_piece.y_0 = self.y_0
        new_piece.rotation = self.rotation
        new_piece.coordinates = self.coordinates
        
        return new_piece

    @property
    def get_self_coords(self):
        return Piece.get_mino_coords(self.x_0, self.y_0, self.rotation, self.type)
    
    @staticmethod
    def get_mino_coords(x_0, y_0, rotation, type):
        coordinate_list = mino_coords_dict[type][rotation]
        # [col, row]
        coordinate_list = [[x_0 + col, y_0 + row] for col, row in coordinate_list]
        
        return coordinate_list