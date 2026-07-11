from ai import *
from util import util_t_spin_board  # star import would make pytest collect util's test_* helpers
from piece import Piece

import numpy as np
import pytest
import ujson

import ai  # module reference for monkeypatching names MCTS/evaluate resolve against

# ------------------------- Helpers -------------------------

def fresh_game():
    game = Game(Config().ruleset)
    game.setup()
    return game

def fake_uniform_evaluate(config, game, network):
    # Stand-in for a network: neutral value, uniform positive policy
    return 0.5, np.ones(POLICY_SHAPE, dtype=np.float32)

def give_o_piece(player):
    # O is its own mirror and has exactly 9 placements on an empty board
    player.piece = Piece(type="O")
    player.piece.move_to_spawn()
    player.held_piece = "O"

def move_matrix_for(grid, piece_type, held_piece):
    # Move matrix for a player with a given board and pieces (brute-force = ground truth)
    game = fresh_game()
    player = game.players[0]
    player.board.grid = [row[:] for row in grid]
    player.piece = Piece(type=piece_type)
    player.piece.move_to_spawn()
    player.held_piece = held_piece
    return get_move_matrix(player, algo='brute-force')

_TINY = {}
def tiny_network():
    # One small keras network shared by the network tests
    if not _TINY:
        config = Config(model='keras', use_tflite=False, blocks=2, pooling_blocks=1,
                        filters=8, cpool=2, o_side_neurons=4, value_head_neurons=4, epochs=1)
        _TINY['config'] = config
        _TINY['model'] = instantiate_network(config, show_summary=False, save_network=False)
    return _TINY['config'], _TINY['model']

# ------------------------- Data loading -------------------------

def test_data_loading(tmp_path, monkeypatch):
    # get_data_filenames picks the newest sets_to_train_with sets, skips junk files;
    # load_data round-trips their contents
    monkeypatch.setattr(Config, "data_dir", property(lambda self: str(tmp_path)))

    sets = {1: [["oldest"]], 2: [["a"], ["b"]], 3: [["newest"]]}
    for number, set in sets.items():
        with open(tmp_path / f"{number}.txt", 'w') as f:
            f.write(ujson.dumps(set))
    (tmp_path / "stats.txt").write_text("not a data set")
    (tmp_path / ".DS_Store").write_text("junk")

    config = Config(sets_to_train_with=2, shuffle=False)

    filenames = get_data_filenames(config)
    assert sorted(filenames) == ["2.txt", "3.txt"]

    data = load_data(config)
    assert sorted(data) == sorted([sets[2], sets[3]])

# ------------------------- MCTS -------------------------

def test_mcts_tree_sanity(monkeypatch):
    # Visit accounting, leaf expansion, move legality, and move selection
    monkeypatch.setattr(ai, "evaluate", fake_uniform_evaluate)

    config = Config(MAX_ITER=25, move_algorithm='brute-force', training=False, visual=False)
    game = fresh_game()

    move, tree, save = MCTS(config, game, None)
    assert save == True # No playout cap randomization outside training

    root = tree.get_node("root")
    children = [tree.get_node(cid) for cid in root.successors(tree.identifier)]

    # Every iteration lands one playout on the root; children got the rest
    assert root.data.visit_count == config.MAX_ITER
    assert sum(child.data.visit_count for child in children) == config.MAX_ITER - 1

    # Root children match the legal move list exactly, and their policy is normalized
    move_matrix = get_move_matrix(game.players[game.turn], algo='brute-force')
    legal_moves = set(m for _, m in get_move_list(move_matrix, np.ones(POLICY_SHAPE)))
    assert set(child.data.move for child in children) == legal_moves
    assert sum(child.data.policy for child in children) == pytest.approx(1)

    # Leaf nodes: only visited nodes get expanded (game copied in, turn flipped)
    for child in children:
        if child.data.visit_count == 0:
            assert child.data.game is None
            assert child.is_leaf()
        else:
            assert child.data.game is not None
            assert child.data.game.turn == 1 - game.turn

    # Not training -> temperature 0 -> the returned move is a most-visited child
    max_visits = max(child.data.visit_count for child in children)
    chosen = [child for child in children if child.data.move == move]
    assert len(chosen) == 1
    assert chosen[0].data.visit_count == max_visits

def test_mcts_terminal_backpropagation(monkeypatch):
    # Forced scenario: every move by player 1 tops them out, so every depth-2 node is
    # a terminal player-0 win. Checks terminal detection and value sign-flipping.
    monkeypatch.setattr(ai, "evaluate", fake_uniform_evaluate)

    config = Config(MAX_ITER=40, move_algorithm='brute-force', training=False, visual=False)
    game = fresh_game()

    give_o_piece(game.players[0])
    give_o_piece(game.players[1])

    # Wall on player 1's board covering the spawn area (rows 4-5), with side gaps so
    # no line can complete: any placement leaves the next spawn blocked -> top-out
    for row in (4, 5):
        for col in range(2, 8):
            game.players[1].board.grid[row][col] = 1
    # Park player 1's active piece below the wall so it starts collision-free
    game.players[1].piece.location.x = 0
    game.players[1].piece.location.y = 10
    game.players[1].piece.coordinates = game.players[1].piece.get_self_coords

    _, tree, _ = MCTS(config, game, None)

    root = tree.get_node("root")
    children = [tree.get_node(cid) for cid in root.successors(tree.identifier)]
    assert len(children) == 9  # O piece on an empty board, hold is also an O

    visited_grandchildren = 0
    revisited_children = 0
    for child in children:
        # First visit backpropagates the fake evaluation (0.5); later visits hit
        # terminal wins (1.0), so the average must rise above 0.5
        if child.data.visit_count == 1:
            assert child.data.value_avg == pytest.approx(0.5)
        elif child.data.visit_count >= 2:
            revisited_children += 1
            assert 0.5 < child.data.value_avg <= config.value_max

        for gc_id in child.successors(tree.identifier):
            grandchild = tree.get_node(gc_id).data
            if grandchild.visit_count > 0:
                visited_grandchildren += 1
                # Player 1 moved and lost: terminal, player 0 wins, mover's value is a loss
                assert grandchild.game.is_terminal == True
                assert grandchild.game.winner == 0
                assert grandchild.value_avg == config.value_min

    assert revisited_children > 0
    assert visited_grandchildren > 0

# ------------------------- get_move_list -------------------------

def test_get_move_list():
    # Formats (policy index, row, col) entries as (policy, (policy_index, col - 2, row))
    move_matrix = np.zeros(POLICY_SHAPE)
    policy = np.zeros(POLICY_SHAPE)

    move_matrix[0][5][0] = 1
    policy[0][5][0] = 0.4
    move_matrix[3][10][7] = 1
    policy[3][10][7] = 0.6
    policy[2][3][3] = 0.9      # policy without a legal move: excluded
    move_matrix[4][4][4] = 1   # legal move with zero policy: excluded by the mask

    move_list = get_move_list(move_matrix, policy)

    assert sorted(move_list) == [(0.4, (0, -2, 5)), (0.6, (3, 5, 10))]

# ------------------------- Network -------------------------

def test_network_instantiation():
    config, model = tiny_network()
    assert len(model.inputs) == 11  # grids/pieces/b2b/combo/garbage x2 + color
    assert model.outputs[0].shape[-1] == 1           # value head
    assert model.outputs[1].shape[-1] == POLICY_SIZE # policy head

def test_network_evaluate():
    config, model = tiny_network()
    value, policy = evaluate(config, fresh_game(), model)

    assert config.value_min <= value <= config.value_max
    assert policy.shape == POLICY_SHAPE
    assert np.all(np.isfinite(policy))
    assert np.sum(policy) == pytest.approx(1, abs=1e-3)  # softmax

def test_network_evaluate_tflite():
    # The tflite path (input reordering, quantization) returns a valid evaluation
    config, model = tiny_network()
    tflite_config = config.copy()
    tflite_config.use_tflite = True

    interpreter = get_interpreter(model)
    value, policy = evaluate(tflite_config, fresh_game(), interpreter)

    assert config.value_min <= value <= config.value_max
    assert policy.shape == POLICY_SHAPE
    assert np.all(np.isfinite(policy))
    assert np.sum(policy) == pytest.approx(1, abs=0.05)  # quantization tolerance

def test_network_training():
    # train_network consumes [features..., value, policy] rows and updates weights
    config, model = tiny_network()
    game = fresh_game()

    row = []
    for feature in game_to_X(game):
        row.append(feature.tolist() if isinstance(feature, np.ndarray) else feature)
    row.append(0.5)                                        # value target
    row.append((np.ones(POLICY_SHAPE) / POLICY_SIZE).tolist())  # policy target

    set = [row[:] for _ in range(8)]

    before = [w.numpy().copy() for w in model.trainable_weights]
    train_network(config, model, set)
    after = [w.numpy() for w in model.trainable_weights]

    assert any(not np.allclose(b, a) for b, a in zip(before, after))

# ------------------------- search_statistics -------------------------

def test_search_statistics():
    # Visit counts become probabilities at [policy_index][row][col + 2]
    tree = treelib.Tree()
    tree.create_node(identifier="root", data=NodeState())

    for move, visits in [((0, -2, 5), 6), ((5, 8, 10), 3), ((18, 0, 0), 1), ((1, 0, 0), 0)]:
        state = NodeState(move=move)
        state.visit_count = visits
        tree.create_node(parent="root", data=state)

    matrix = search_statistics(tree)

    assert matrix[0][5][0] == 0.6    # col -2 -> buffer index 0
    assert matrix[5][10][10] == 0.3
    assert matrix[18][0][2] == 0.1
    assert matrix[1][0][2] == 0      # unvisited move excluded
    assert np.sum(matrix) == pytest.approx(1)
    assert np.count_nonzero(matrix) == 3

# ------------------------- Game data extraction -------------------------

def test_get_grids():
    game = fresh_game()
    game.players[0].board.grid[25][0] = 'T'
    game.players[1].board.grid[25][9] = 'I'

    grids = get_grids(game)
    assert grids[0][25][0] == 1  # minos simplified to 1s
    assert grids[1][25][9] == 1
    assert game.players[0].board.grid[25][0] == 'T'  # original not mutated

    game.turn = 1  # active player first
    assert get_grids(game)[0][25][9] == 1

def test_get_pieces():
    game = fresh_game()
    game.players[0].held_piece = 'T'

    pieces = get_pieces(game)
    assert pieces.shape == (2, 2 + PREVIEWS, len(MINOS))
    for i, player in enumerate(game.players):
        assert pieces[i][0][MINOS.index(player.piece.type)] == 1  # active piece one-hot
        assert pieces[i][0].sum() == 1
        for j, queue_piece in enumerate(player.queue.pieces[:PREVIEWS]):
            assert pieces[i][j + 2][MINOS.index(queue_piece)] == 1
    assert pieces[0][1][MINOS.index('T')] == 1  # held piece
    assert pieces[1][1].sum() == 0              # no held piece

    game.turn = 1
    assert np.array_equal(get_pieces(game)[0], pieces[1])

def test_get_stat():
    game = fresh_game()
    game.players[0].stats.b2b = 3
    game.players[1].stats.b2b = 7

    assert get_stat(game, 'b2b') == [3, 7]
    game.turn = 1
    assert get_stat(game, 'b2b') == [7, 3]

def test_get_garbage():
    game = fresh_game()
    game.players[0].garbage_to_receive = [1, 2, 3]

    assert get_garbage(game) == [3, 0]
    game.turn = 1
    assert get_garbage(game) == [0, 3]

def test_game_to_X():
    game = fresh_game()
    game.players[0].stats.b2b = 2
    game.players[1].stats.combo = 4

    X = game_to_X(game)
    assert len(X) == 11
    assert X[0] == get_grids(game)[0]
    assert np.array_equal(X[1], get_pieces(game)[0])
    assert X[2] == 2   # active b2b
    assert X[8] == 4   # opponent combo
    assert X[10] == 0  # player 0's color

    game.turn = 1
    X = game_to_X(game)
    assert X[7] == 2   # players swapped
    assert X[3] == 4
    assert X[10] == 1

# ------------------------- Reflections -------------------------

def test_reflect_grid():
    grid = [[1, 2, 3], [4, 5, 6]]
    assert reflect_grid(grid) == [[3, 2, 1], [6, 5, 4]]
    assert reflect_grid(reflect_grid(grid)) == grid

def test_reflect_pieces():
    # MINOS = "ZLOSIJT": Z<->S, L<->J, O/I/T unchanged
    piece_table = np.zeros((2 + PREVIEWS, len(MINOS)), dtype=int)
    for row, piece in enumerate("ZLOSIJT"):
        piece_table[row][MINOS.index(piece)] = 1

    reflected = reflect_pieces(piece_table)
    swap = {"Z": "S", "S": "Z", "L": "J", "J": "L"}
    for row, piece in enumerate("ZLOSIJT"):
        expected = swap.get(piece, piece)
        assert reflected[row][MINOS.index(expected)] == 1
        assert reflected[row].sum() == 1

    assert np.array_equal(reflect_pieces(reflect_pieces(piece_table)), piece_table)

def test_reflect_policy_double_reflection():
    # Reflecting twice returns the original, including T (spin indices) and I
    # (rotation 1<->3 column adjustment)
    for piece_type, held in [("T", "T"), ("I", "I")]:
        matrix = move_matrix_for(util_t_spin_board, piece_type, held)
        double = np.array(reflect_policy(reflect_policy(matrix)), dtype=float)
        assert np.array_equal(double.astype(bool), matrix.astype(bool))

def test_reflect_policy_matches_mirrored_board():
    # The reflected policy must equal the move matrix generated on the mirrored board.
    # T is its own mirror; Z/L on the original correspond to S/J on the mirror.
    for pieces, mirrored_pieces in [(("T", "T"), ("T", "T")), (("Z", "L"), ("S", "J"))]:
        matrix = move_matrix_for(util_t_spin_board, *pieces)
        mirrored_matrix = move_matrix_for(reflect_grid(util_t_spin_board), *mirrored_pieces)

        reflected = np.array(reflect_policy(matrix), dtype=float)
        assert np.array_equal(reflected.astype(bool), mirrored_matrix.astype(bool))

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

# ------------------------- battle_networks -------------------------

def test_battle_networks_side_switching(monkeypatch):
    # Sentinel "networks" and a fake MCTS: network "A" always wins by making its
    # opponent top out. Verifies who plays which color each game and that wins are
    # attributed to the right network when colors flip.
    calls = []

    def fake_mcts(config, game, network):
        calls.append((game.turn, network))
        if network == "A":
            game.players[1 - game.turn].game_over = True
        # A legal harddrop of the active piece (rotation 0 always fits at col 3)
        piece_type = game.players[game.turn].piece.type
        return (policy_piece_to_index[piece_type][0][0], 3, 10), None, False

    monkeypatch.setattr(ai, "MCTS", fake_mcts)
    config = Config(visual=False)

    # A as network 1: must be credited with every win, on both colors
    wins, result = battle_networks("A", config, "B", config, None, None, 4)
    assert wins.tolist() == [4, 0]
    assert result is None  # no threshold given
    # Even games: A plays player 0 and moves first. Odd games: colors flip, so B
    # (as player 0) moves first and A finishes as player 1.
    assert calls == [(0, "A"), (0, "B"), (1, "A"), (0, "A"), (0, "B"), (1, "A")]

    # A as network 2: same games, wins now credited to the other slot
    calls.clear()
    wins, result = battle_networks("B", config, "A", config, None, None, 4)
    assert wins.tolist() == [0, 4]
    assert calls == [(0, "B"), (1, "A"), (0, "A"), (0, "B"), (1, "A"), (0, "A")]

def test_battle_networks_threshold(monkeypatch):
    # Early termination once the threshold is unreachable/reached
    def fake_mcts(config, game, network):
        if network == "A":
            game.players[1 - game.turn].game_over = True
        piece_type = game.players[game.turn].piece.type
        return (policy_piece_to_index[piece_type][0][0], 3, 10), None, False

    monkeypatch.setattr(ai, "MCTS", fake_mcts)
    config = Config(visual=False)

    wins, result = battle_networks("A", config, "B", config, 0.75, 'moreorequal', 4)
    assert result is True
    assert wins.tolist() == [3, 0]  # stopped as soon as 3 >= 0.75 * 4

# pytest tests.py
if __name__ == "__main__":
    test_reflections()
