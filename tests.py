from ai import *
from util import *

def test_reflections():
    # Tests that double reflections return the original grid, pieces, and policy.
    c = Config()
    interpreter = get_interpreter(load_best_model(c))

    grid = [x[:] for x in util_t_spin_board] # copy
    game = Game(c.ruleset)
    game.setup()
    game.players[game.turn].board.grid = grid
    pieces = get_pieces(game)[0]
    # _, policy = evaluate(c, game, interpreter)
    move, tree, save = MCTS(c, game, interpreter)
    search_matrix = search_statistics(tree)

    assert all([r == n for (rl, nl) in zip(reflect_grid(reflect_grid(grid)), grid) for (r, n) in zip(rl, nl)]) # Grid
    assert all([r == n for (rl, nl) in zip(reflect_pieces(reflect_pieces(pieces)), pieces) for (r, n) in zip(rl, nl)]) # Pieces
    assert all([r == n for (rl, nl) in zip(reflect_policy(reflect_policy(search_matrix)), search_matrix) for (r, n) in zip(rl, nl)]) # Policy

# pytest tests.py::test_reflections
if __name__ == "__main__":
    test_reflections()