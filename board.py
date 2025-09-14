from const import *
import numpy as np

class Board():

    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=np.int8)

    def empty_line(self, row):
        self.grid[row, :] = 0
        
    def create_garbage(self, garbage):
        garbage_lines = len(garbage)
        
        # Replace each row with the row (n garbage) below
        self.grid[:-garbage_lines] = self.grid[garbage_lines:]

        # Spawn in garbage
        self.grid[-garbage_lines:] = 1

        # Create holes in garbage
        for i, garbage_col in enumerate(garbage):
            self.grid[ROWS - garbage_lines + i, garbage_col] = 0

    def copy(self):
        new_board = Board()
        new_board.grid = self.grid.copy()
        return new_board