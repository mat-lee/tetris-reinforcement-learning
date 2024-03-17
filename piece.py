from const import *

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

    def move_to_spawn(self):
        # Move a newly created "O" piece to the right
        displacement = (1 if self.type == "O" else 0)

        self.x_0 = 3 + displacement
        self.y_0 = 0

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