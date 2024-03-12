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
    
    def garbage_line(self, spawn_col):
        # Replace each row with the row below
        for row in range(ROWS - 1):
            for col in range(COLS):
                self.grid[row][col].type = self.grid[row + 1][col].type

            # Create row of garbage
        for col in range(COLS):
            if spawn_col != col:
                self.grid[ROWS - 1][col].type = "garbage"
            self.grid[ROWS - 1][spawn_col].type = "empty"