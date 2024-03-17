from const import *
from mino import Mino

class Board():

    def __init__(self):
        self.grid = [[0]* COLS for row in range(ROWS)]

        for row in range(ROWS):
            for col in range(COLS):
                self.grid[row][col] = Mino(row, col)

    def empty_line(self, row):
        for col in range(COLS):
            self.grid[row][col] = Mino(row, col)
    
    def create_garbage(self, garbage):
        garbage_lines = len(garbage)
        
        # Replace each row with the row (n garbage) below
        for row in range(ROWS - garbage_lines):
            for col in range(COLS):
                self.grid[row][col].type = self.grid[row + garbage_lines][col].type
        
        # Spawn in garbage
        for col in range(COLS):
            for i, garbage_col in enumerate(garbage):
                if garbage_col != col:
                    self.grid[ROWS - garbage_lines + i][col].type = "garbage"
                self.grid[ROWS - garbage_lines + i][garbage_col].type = "empty"