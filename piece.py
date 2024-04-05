from const import *

import copy

class Piece:
    '''
    The piece matrix and the center of the piece have a location attribute
    The coordinates are saved and the piece is drawn on top
    '''

    def __init__(self, matrix, type=None, x_0=0, y_0=0):
        # Coords describe the row of the top left value in the matrix
        self.x_0 = x_0
        self.y_0 = y_0
        self.type = type
        self.matrix = matrix
        self.rotation = 0 # 1 is right, 2 is 180, 3 is left

    def get_spawn_location(self):
        # Move a newly created "O" piece to the right
        displacement = (1 if self.type == "O" else 0)

        return (3 + displacement, 0)

    def move_to_spawn(self):
        self.x_0, self.y_0 = self.get_spawn_location()

    def rotated_matrix(self, initial, final):
        diff = (final - initial) % 4

        if diff == 0:
            return(self.matrix)

        # Clockwise rotation
        if diff == 1:
            return(list(zip(*self.matrix[::-1])))

        # 180 rotation
        if diff == 2:
            return([x[::-1] for x in self.matrix[::-1]])

        # Counter-clockwise rotation
        if diff == 3:
            return(list(zip(*self.matrix))[::-1])
    
    def update_rotation(self):
        self.matrix = copy.deepcopy(piece_dict[self.type])
        self.matrix = self.rotated_matrix(0, self.rotation)

    @property
    def coords(self):
        return(Piece.get_mino_coords(self.x_0, self.y_0, self.matrix))
    
    @staticmethod
    def get_mino_coords(x_0, y_0, piece_matrix):
        coordinate_list = []
        cols = len(piece_matrix[0])
        rows = len(piece_matrix)
        for x in range(cols):
            for y in range(rows):
                if piece_matrix[y][x] != 0:
                    col = x_0 + x
                    row = y_0 + y
                    coordinate_list.append([col, row])
        
        return(coordinate_list)