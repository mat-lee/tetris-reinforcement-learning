from ai import *
from util import *

def _get_move_matrix_for_board(board, piece_type, algo):
    game = Game(ruleset='s1')
    game.setup()
    game.players[game.turn].board.grid = [row[:] for row in board]
    game.players[game.turn].piece = Piece(type=piece_type)
    game.players[game.turn].piece.move_to_spawn()
    return get_move_matrix(game.players[game.turn], algo=algo)


def test_brute_force_soundness():
    # Verify brute-force finds at least 1 move and at most policy capacity on real boards
    test_boards = [util_t_spin_board, util_z_spin_board, util_move_algo_board, util_move_algo_board_2]
    for board in test_boards:
        for piece_type in MINOS:
            matrix = _get_move_matrix_for_board(board, piece_type, 'brute-force')
            n_moves = int(np.sum(matrix))
            assert n_moves >= 1, f"brute-force found 0 moves for piece {piece_type}"
            assert n_moves <= matrix.size, f"brute-force found more moves than policy capacity for piece {piece_type}"

    # On a fully empty board, O piece should find exactly 9 positions (columns 0-8, single rotation).
    # Set held_piece='O' so the generator deduplicates and only runs for the O piece.
    empty_board = [[0] * COLS for _ in range(ROWS)]
    game = Game(ruleset='s1')
    game.setup()
    game.players[game.turn].board.grid = empty_board
    game.players[game.turn].piece = Piece(type='O')
    game.players[game.turn].piece.move_to_spawn()
    game.players[game.turn].held_piece = 'O'
    o_matrix = get_move_matrix(game.players[game.turn], algo='brute-force')
    assert int(np.sum(o_matrix)) == 9, f"Expected 9 O-piece placements on empty board, got {int(np.sum(o_matrix))}"


def _make_player(board, piece_type):
    """Build a player with the given board and piece. Use the same instance for both algorithm calls."""
    game = Game(ruleset='s1')
    game.setup()
    game.players[game.turn].board.grid = [row[:] for row in board]
    game.players[game.turn].piece = Piece(type=piece_type)
    game.players[game.turn].piece.move_to_spawn()
    return game.players[game.turn]


def test_convolutional_matches_brute_force():
    # Verify convolutional algorithm produces identical results to brute-force.
    # Uses the same player instance for both calls so the held piece is identical.
    test_boards = [util_t_spin_board, util_z_spin_board, util_move_algo_board, util_move_algo_board_2]
    for board in test_boards:
        for piece_type in MINOS:
            player = _make_player(board, piece_type)
            bf_matrix = get_move_matrix(player, algo='brute-force')
            conv_matrix = get_move_matrix(player, algo='convolutional')
            assert np.array_equal(bf_matrix, conv_matrix), (
                f"convolutional != brute-force for piece {piece_type}\n"
                f"  missing: {np.argwhere(bf_matrix & ~conv_matrix).tolist()}\n"
                f"  extra:   {np.argwhere(conv_matrix & ~bf_matrix).tolist()}"
            )


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