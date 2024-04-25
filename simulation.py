from ai import MCTS
from game import Game

# Simulate playing the ai against itself to generate data

# Policy: 25 x 11 x 4 x 2 = 2376
# Rows x Columns x Rotations x Hold

# Terminate games that extend too long

# Having two AI's play against each other.
# At each move, save X and y
# X: Active board + Opponent board
# y: MCTS search stats + Whether they won or not

class Simulation():
    def __init__(self) -> None:
        game = Game()
        dataset = []
    
    def play_game(self):
        while True:
            move, tree = MCTS(self.game)
            self.game.make_move(move)

            