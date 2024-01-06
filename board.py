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