from const import *

class Piece:
    '''
    The piece matrix and the center of the piece have a location attribute
    The coordinates are saved and the piece is drawn on top
    '''

    def __init__(self, matrix, type=None, x_0=0, y_0=0):
        self.x_0 = x_0
        self.y_0 = y_0
        self.type = type
        self.matrix = matrix
        self.rotation = 0 # 1 is right, 2 is 180, 3 is left

    def move_to_spawn(self):
        # Move a newly created "O" piece to the right
        displacement = (1 if self.type == "O" else 0)

        self.x_0 = 3 + displacement
        self.y_0 = 0

    def get_mino_coords(self):
        coordinate_list = []
        cols = len(self.matrix[0])
        rows = len(self.matrix)
        for x in range(cols):
            for y in range(rows):
                if self.matrix[y][x] != 0:
                    col = self.x_0 + x
                    row = self.y_0 + y
                    coordinate_list.append([col, row])
        
        return(coordinate_list)