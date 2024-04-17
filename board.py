from const import *

class Board():

    def __init__(self):
        self.grid = [[0 for x in range(COLS)] for y in range(ROWS)]

    def empty_line(self, row):
        for col in range(COLS):
            self.grid[row][col] = 0
    
    def create_garbage(self, garbage):
        garbage_lines = len(garbage)
        
        # Replace each row with the row (n garbage) below
        for row in range(ROWS - garbage_lines):
            for col in range(COLS):
                self.grid[row][col] = self.grid[row + garbage_lines][col]
        
        # Spawn in garbage
        for col in range(COLS):
            for i, garbage_col in enumerate(garbage):
                if garbage_col != col:
                    self.grid[ROWS - garbage_lines + i][col] = 1
                self.grid[ROWS - garbage_lines + i][garbage_col] = 0

    def copy(self):
        new_board = Board()
        new_board.grid = [x[:] for x in self.grid]
        return new_board