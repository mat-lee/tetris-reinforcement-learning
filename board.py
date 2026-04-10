import numpy as np
from const import ROWS, COLS

class Board():

    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=object)

    def empty_line(self, row):
        self.grid[row, :] = 0

    def create_garbage(self, garbage):
        garbage_lines = len(garbage)

        # Shift rows up (discard top garbage_lines rows)
        self.grid[:ROWS - garbage_lines] = self.grid[garbage_lines:]

        # Spawn in garbage rows at the bottom
        for i, garbage_col in enumerate(garbage):
            self.grid[ROWS - garbage_lines + i, :] = 1
            self.grid[ROWS - garbage_lines + i, garbage_col] = 0

    def copy(self):
        new = Board.__new__(Board)
        new.grid = self.grid.copy()
        return new
