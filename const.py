import pygame
import numpy as np
from numpy import prod

# Board Dimensions:
# TETR.IO standard: 10 wide, 40 tall (20 visible + 20 hidden buffer/vanish zone).
ROWS = 40
SPAWN_ROW = 23
GRID_ROWS = 20
COLS = 10

# AI settings
MAX_MOVES = 1000

# Controls:
k_move_left    = pygame.K_LEFT
k_move_right   = pygame.K_RIGHT
k_soft_drop    = pygame.K_DOWN
k_hard_drop    = pygame.K_SPACE
k_make_ai_move = pygame.K_f
k_rotate_ccw   = pygame.K_z
k_rotate_cw    = pygame.K_UP
k_rotate_180   = pygame.K_a
k_hold         = pygame.K_c

k_undo         = pygame.K_1
k_redo         = pygame.K_2
k_restart      = pygame.K_r
k_add_garbage  = pygame.K_g
k_switch       = pygame.K_s
k_print_board  = pygame.K_p

# graphics break when ARR = 0
# idc
DAS = 100
ARR = 0
SDF = 1/10

PREVIEWS = 5

# Screen Dimensions:
MINO_SIZE = 20

N_BUFFER = 0
S_BUFFER = 0
E_BUFFER = 0
W_BUFFER = 0

HOLD_WIDTH = 5 * MINO_SIZE
QUEUE_WIDTH = 5 * MINO_SIZE

WIDTH = 2 * (COLS * MINO_SIZE + HOLD_WIDTH + QUEUE_WIDTH + E_BUFFER + W_BUFFER)
HEIGHT = ROWS * MINO_SIZE + N_BUFFER + S_BUFFER

# Stats locations
STAT_SETTINGS = [
    {'text': "attack_text", 'size': int(MINO_SIZE * 0.65), 'location': (0                           , 4  * MINO_SIZE)},
    {'text': "B2B",         'size': int(MINO_SIZE * 1.0 ), 'location': (0                           , 7  * MINO_SIZE)},
    {'text': "COMBO",       'size': int(MINO_SIZE * 1.0 ), 'location': (0                           , 9  * MINO_SIZE)},
    {'text': "LOSES",       'size': int(MINO_SIZE * 0.8 ), 'location': (0                           , 14 * MINO_SIZE)},
    {'text': "PIECES",      'size': int(MINO_SIZE * 0.8 ), 'location': (0                           , 16 * MINO_SIZE)},
    {'text': "pieces_stat", 'size': int(MINO_SIZE * 0.9 ), 'location': (HOLD_WIDTH - 1.5 * MINO_SIZE, 16 * MINO_SIZE)},
    {'text': "ATTACK",      'size': int(MINO_SIZE * 0.8 ), 'location': (0                           , 18 * MINO_SIZE)},
    {'text': "attack_stat", 'size': int(MINO_SIZE * 0.9 ), 'location': (HOLD_WIDTH - 1.5 * MINO_SIZE, 18 * MINO_SIZE)},
    {'text': "EVAL",        'size': int(MINO_SIZE * 0.8 ), 'location': (0                           , 19 * MINO_SIZE)},
    {'text': "value_stat",  'size': int(MINO_SIZE * 0.9 ), 'location': (HOLD_WIDTH - 1.5 * MINO_SIZE, 19 * MINO_SIZE)}
]

MINOS = "ZLOSIJT"
MINO_TO_INDEX = {m: i for i, m in enumerate(MINOS)}

# Piece Matrices:
# For encoding AI information
policy_pieces = {
    "O": [0],
    "Z": [0, 1],
    "S": [0, 1],
    "I": [0, 1],
    "L": [0, 1, 2, 3],
    "J": [0, 1, 2, 3],
    "T": [0, 1, 2, 3]
}

policy_index_to_piece = {
    0: ["O", 0, 0],

    1: ["Z", 0, 0],
    2: ["Z", 1, 0],

    3: ["S", 0, 0],
    4: ["S", 1, 0],

    5: ["I", 0, 0],
    6: ["I", 1, 0],

    7: ["L", 0, 0],
    8: ["L", 1, 0],
    9: ["L", 2, 0],
    10: ["L", 3, 0],

    11: ["J", 0, 0],
    12: ["J", 1, 0],
    13: ["J", 2, 0],
    14: ["J", 3, 0],
    
    15: ["T", 0, 0],
    16: ["T", 1, 0],
    17: ["T", 2, 0],
    18: ["T", 3, 0],

    19: ["T", 0, 1],
    20: ["T", 1, 1],
    21: ["T", 2, 1],
    22: ["T", 3, 1],

    23: ["T", 0, 2],
    24: ["T", 1, 2],
    25: ["T", 2, 2],
    26: ["T", 3, 2],
}

# Leftside buffer is 2
# Can't place piece at the bottom most row
POLICY_SHAPE = (len(policy_index_to_piece), ROWS - 1, COLS + 2 - 1)
POLICY_SIZE = prod(POLICY_SHAPE)

policy_piece_to_index = {
    "O": {0: {0: 0}},
    "Z": {0: {0: 1}, 1: {0: 2}},
    "S": {0: {0: 3}, 1: {0: 4}},
    "I": {0: {0: 5}, 1: {0: 6}},
    "L": {0: {0: 7}, 1: {0: 8}, 2: {0: 9}, 3: {0: 10}},
    "J": {0: {0: 11}, 1: {0: 12}, 2: {0: 13}, 3: {0: 14}},
    "T": {0: {0: 15, 1: 19, 2: 23}, 
          1: {0: 16, 1: 20, 2: 24}, 
          2: {0: 17, 1: 21, 2: 25}, 
          3: {0: 18, 1: 22, 2: 26}},
}

piece_dict = {
    "Z": [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0]
    ],
    "L": [
        [0, 0, 2],
        [2, 2, 2],
        [0, 0, 0]
    ],
    "O": [
        [3, 3],
        [3, 3]
    ],
    "S": [
        [0, 4, 4],
        [4, 4, 0],
        [0, 0, 0]]
    ,
    "I": [
        [0, 0, 0, 0],
        [5, 5, 5, 5],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ],
    "J": [
        [6, 0, 0,],
        [6, 6, 6],
        [0, 0, 0]
    ],
    "T": [
        [0, 7, 0],
        [7, 7, 7],
        [0, 0, 0]
    ]
}

# Color values:
color_dict = {
    0:         (0,   0,   0),   # Empty
    1:         (85,  85, 85),   # Garbage
    "ghost":   (32,  32,  32),  # Ghost piece
    "Z":       (255, 1,   0),   #1
    "L":       (254, 170, 0),   #2
    "O":       (255, 254, 2),   #3
    "S":       (0,   234, 1),   #4
    "I":       (0,   211, 255), #5
    "J":       (0,   0,   255), #6
    "T":       (170, 0,   254)  #7
}

# Wallkick tables
wallkicks = {
    0: {
        1: [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
        3: [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
        2: [[0, 0], [0, 1], [1, 1], [-1, 1], [1, 0], [-1, 0]]
    },
    1: {
        0: [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
        2: [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
        3: [[0, 0], [1, 0], [1, 2], [1, 1], [0, 2], [0, 1]]
    },
    2: {
        1: [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
        3: [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
        0: [[0, 0], [0, -1], [-1, -1], [1, -1], [-1, 0], [1, 0]]
    },
    3: {
        2: [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
        0: [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
        1: [[0, 0], [-1, 0], [-1, 2], [-1, 1], [0, 2], [0, 1]]
    },
}

i_wallkicks = {
    0: {
        1: [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
        3: [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
        2: [[0, 0], [0, 1]]
    },
    1: {
        0: [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
        2: [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
        3: [[0, 0], [1, 0]]
    },
    2: {
        1: [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
        3: [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
        0: [[0, 0], [0, -1]]
    },
    3: {
        2: [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
        0: [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
        1: [[0, 0], [-1 ,0]]
    },
}

# Returns the coordinates of each piece/rotation at (0, 0)
mino_coords_dict = {
    "Z": {
        0: [[0, 0], [1, 0], [1, 1], [2, 1]],
        1: [[1, 1], [1, 2], [2, 0], [2, 1]],
        2: [[0, 1], [1, 1], [1, 2], [2, 2]],
        3: [[0, 1], [0, 2], [1, 0], [1, 1]],
    },
    "L": {
        0: [[0, 1], [1, 1], [2, 0], [2, 1]],
        1: [[1, 0], [1, 1], [1, 2], [2, 2]],
        2: [[0, 1], [0, 2], [1, 1], [2, 1]],
        3: [[0, 0], [1, 0], [1, 1], [1, 2]],
    },
    "O": {
        0: [[0, 0], [0, 1], [1, 0], [1, 1]],
        1: [[0, 0], [0, 1], [1, 0], [1, 1]],
        2: [[0, 0], [0, 1], [1, 0], [1, 1]],
        3: [[0, 0], [0, 1], [1, 0], [1, 1]],
    },
    "S": {
        0: [[0, 1], [1, 0], [1, 1], [2, 0]],
        1: [[1, 0], [1, 1], [2, 1], [2, 2]],
        2: [[0, 2], [1, 1], [1, 2], [2, 1]],
        3: [[0, 0], [0, 1], [1, 1], [1, 2]],
    },
    "I": {
        0: [[0, 1], [1, 1], [2, 1], [3, 1]],
        1: [[2, 0], [2, 1], [2, 2], [2, 3]],
        2: [[0, 2], [1, 2], [2, 2], [3, 2]],
        3: [[1, 0], [1, 1], [1, 2], [1, 3]],
    },
    "J": {
        0: [[0, 0], [0, 1], [1, 1], [2, 1]],
        1: [[1, 0], [1, 1], [1, 2], [2, 0]],
        2: [[0, 1], [1, 1], [2, 1], [2, 2]],
        3: [[0, 2], [1, 0], [1, 1], [1, 2]],
    },
    "T": {
        0: [[0, 1], [1, 0], [1, 1], [2, 1]],
        1: [[1, 0], [1, 1], [1, 2], [2, 1]],
        2: [[0, 1], [1, 1], [1, 2], [2, 1]],
        3: [[0, 1], [1, 0], [1, 1], [1, 2]],
    },
}

# For generating moves quicker
piece_hover_coordinates = {
    "Z": [
        (3, 3, 0),
        (2, 3, 0),
        (1, 3, 0),
        (0, 3, 0),
        (4, 3, 0),
        (5, 3, 0),
        (6, 3, 0),
        (7, 3, 0),
        (3, 3, 1),
        (2, 3, 1),
        (1, 3, 1),
        (0, 3, 1),
        (-1, 3, 1),
        (4, 3, 1),
        (5, 3, 1),
        (6, 3, 1),
        (7, 3, 1),
        (3, 3, 2),
        (2, 3, 2),
        (1, 3, 2),
        (0, 3, 2),
        (4, 3, 2),
        (5, 3, 2),
        (6, 3, 2),
        (7, 3, 2),
        (3, 3, 3),
        (2, 3, 3),
        (1, 3, 3),
        (0, 3, 3),
        (4, 3, 3),
        (5, 3, 3),
        (6, 3, 3),
        (7, 3, 3),
        (8, 3, 3),
    ],
    "L": [
        (3, 3, 0),
        (2, 3, 0),
        (1, 3, 0),
        (0, 3, 0),
        (4, 3, 0),
        (5, 3, 0),
        (6, 3, 0),
        (7, 3, 0),
        (3, 3, 1),
        (2, 3, 1),
        (1, 3, 1),
        (0, 3, 1),
        (-1, 3, 1),
        (4, 3, 1),
        (5, 3, 1),
        (6, 3, 1),
        (7, 3, 1),
        (3, 3, 2),
        (2, 3, 2),
        (1, 3, 2),
        (0, 3, 2),
        (4, 3, 2),
        (5, 3, 2),
        (6, 3, 2),
        (7, 3, 2),
        (3, 3, 3),
        (2, 3, 3),
        (1, 3, 3),
        (0, 3, 3),
        (4, 3, 3),
        (5, 3, 3),
        (6, 3, 3),
        (7, 3, 3),
        (8, 3, 3),
    ],
    "O": [
        (4, 3, 0),
        (3, 3, 0),
        (2, 3, 0),
        (1, 3, 0),
        (0, 3, 0),
        (5, 3, 0),
        (6, 3, 0),
        (7, 3, 0),
        (8, 3, 0),
    ],
    "S": [
        (3, 3, 0),
        (2, 3, 0),
        (1, 3, 0),
        (0, 3, 0),
        (4, 3, 0),
        (5, 3, 0),
        (6, 3, 0),
        (7, 3, 0),
        (3, 3, 1),
        (2, 3, 1),
        (1, 3, 1),
        (0, 3, 1),
        (-1, 3, 1),
        (4, 3, 1),
        (5, 3, 1),
        (6, 3, 1),
        (7, 3, 1),
        (3, 3, 2),
        (2, 3, 2),
        (1, 3, 2),
        (0, 3, 2),
        (4, 3, 2),
        (5, 3, 2),
        (6, 3, 2),
        (7, 3, 2),
        (3, 3, 3),
        (2, 3, 3),
        (1, 3, 3),
        (0, 3, 3),
        (4, 3, 3),
        (5, 3, 3),
        (6, 3, 3),
        (7, 3, 3),
        (8, 3, 3),
    ],
    "I": [
        (3, 3, 0),
        (2, 3, 0),
        (1, 3, 0),
        (0, 3, 0),
        (4, 3, 0),
        (5, 3, 0),
        (6, 3, 0),
        (3, 3, 1),
        (2, 3, 1),
        (1, 3, 1),
        (0, 3, 1),
        (-1, 3, 1),
        (-2, 3, 1),
        (4, 3, 1),
        (5, 3, 1),
        (6, 3, 1),
        (7, 3, 1),
        (3, 3, 2),
        (2, 3, 2),
        (1, 3, 2),
        (0, 3, 2),
        (4, 3, 2),
        (5, 3, 2),
        (6, 3, 2),
        (3, 3, 3),
        (2, 3, 3),
        (1, 3, 3),
        (0, 3, 3),
        (-1, 3, 3),
        (4, 3, 3),
        (5, 3, 3),
        (6, 3, 3),
        (7, 3, 3),
        (8, 3, 3),
    ],
    "J": [
        (3, 3, 0),
        (2, 3, 0),
        (1, 3, 0),
        (0, 3, 0),
        (4, 3, 0),
        (5, 3, 0),
        (6, 3, 0),
        (7, 3, 0),
        (3, 3, 1),
        (2, 3, 1),
        (1, 3, 1),
        (0, 3, 1),
        (-1, 3, 1),
        (4, 3, 1),
        (5, 3, 1),
        (6, 3, 1),
        (7, 3, 1),
        (3, 3, 2),
        (2, 3, 2),
        (1, 3, 2),
        (0, 3, 2),
        (4, 3, 2),
        (5, 3, 2),
        (6, 3, 2),
        (7, 3, 2),
        (3, 3, 3),
        (2, 3, 3),
        (1, 3, 3),
        (0, 3, 3),
        (4, 3, 3),
        (5, 3, 3),
        (6, 3, 3),
        (7, 3, 3),
        (8, 3, 3),
    ],
    "T": [
        (3, 3, 0),
        (2, 3, 0),
        (1, 3, 0),
        (0, 3, 0),
        (4, 3, 0),
        (5, 3, 0),
        (6, 3, 0),
        (7, 3, 0),
        (3, 3, 1),
        (2, 3, 1),
        (1, 3, 1),
        (0, 3, 1),
        (-1, 3, 1),
        (4, 3, 1),
        (5, 3, 1),
        (6, 3, 1),
        (7, 3, 1),
        (3, 3, 2),
        (2, 3, 2),
        (1, 3, 2),
        (0, 3, 2),
        (4, 3, 2),
        (5, 3, 2),
        (6, 3, 2),
        (7, 3, 2),
        (3, 3, 3),
        (2, 3, 3),
        (1, 3, 3),
        (0, 3, 3),
        (4, 3, 3),
        (5, 3, 3),
        (6, 3, 3),
        (7, 3, 3),
        (8, 3, 3),
    ],
}

### New Policy Format

policy_piece_grids_no_padding = {}

def rotate_grid_clockwise(grid):
    return np.rot90(grid, k=-1)

def strip_zero_padding(grid):
    rows = np.any(grid, axis=1)
    cols = np.any(grid, axis=0)
    return grid[np.ix_(rows, cols)]

for policy_index in policy_index_to_piece:
    piece, rotation, t_spin_index = policy_index_to_piece[policy_index]
    grid = np.array(piece_dict[piece])
    for _ in range(rotation):
        grid = rotate_grid_clockwise(grid)

    policy_piece_grids_no_padding[policy_index] = strip_zero_padding(grid)

# Per-policy-index buffers. Both legacy index (ri, ci) and spatial index
# (new_row, new_col) map to a common reference coordinate — the rotated PADDED
# bbox's top-left in board frame — via their respective buffers:
#   ri      + LEGACY_POLICY_ROW_BUFFER = padded_top_row
#   ci      + LEGACY_POLICY_COL_BUFFER = padded_top_col
#   new_row + policy_row_buffer[pi]    = padded_top_row
#   new_col + policy_col_buffer[pi]    = padded_top_col
# Since stripped_top = padded_top + pad_offset and new_(row,col) = stripped_top,
# the new-frame buffer is -pad_offset. Conversion goes through the common ref:
#   new_row = (ri + LEGACY_POLICY_ROW_BUFFER) - policy_row_buffer[pi]
#   new_col = (ci + LEGACY_POLICY_COL_BUFFER) - policy_col_buffer[pi]
LEGACY_POLICY_ROW_BUFFER = 0    # ri IS the padded-bbox top row.
LEGACY_POLICY_COL_BUFFER = -2   # ci = padded-bbox top col + 2.

policy_row_buffer = {}
policy_col_buffer = {}
policy_row_range = {}  # (min, max) inclusive, valid new_row range per policy_index
policy_col_range = {}

for policy_index in policy_index_to_piece:
    piece, rotation, _ = policy_index_to_piece[policy_index]
    padded = np.array(piece_dict[piece])
    for _ in range(rotation):
        padded = rotate_grid_clockwise(padded)
    pad_offset_row = int(np.argmax(np.any(padded, axis=1)))
    pad_offset_col = int(np.argmax(np.any(padded, axis=0)))
    policy_row_buffer[policy_index] = -pad_offset_row
    policy_col_buffer[policy_index] = -pad_offset_col

    kh, kw = policy_piece_grids_no_padding[policy_index].shape
    policy_row_range[policy_index] = (0, ROWS - kh)
    policy_col_range[policy_index] = (0, COLS - kw)

# Vectorized copies of the buffers/ranges above, indexed by policy_index, so
# legacy (pi, ri, ci) indices can be converted to spatial (new_row, new_col)
# in bulk without materializing the legacy tensor (see ai.get_move_list).
_policy_indices = sorted(policy_index_to_piece)
POLICY_ROW_BUFFER_ARR = np.array([policy_row_buffer[i] for i in _policy_indices], dtype=np.intp)
POLICY_COL_BUFFER_ARR = np.array([policy_col_buffer[i] for i in _policy_indices], dtype=np.intp)
POLICY_ROW_MAX_ARR = np.array([policy_row_range[i][1] for i in _policy_indices], dtype=np.intp)
POLICY_COL_MAX_ARR = np.array([policy_col_range[i][1] for i in _policy_indices], dtype=np.intp)

# Generate A and M matrices for each policy index.
#
# A_matrices[policy_index] has shape (ROWS, COLS, n): each "kernel" along axis 2 is
# the binary occupancy heatmap of one valid placement of the (stripped) piece grid.
# Forward: heatmap y = A @ w where w is the action weights (length n).
# M_matrices[policy_index] has shape (n, ROWS, COLS): rows of the Moore-Penrose
# pseudoinverse of A.reshape(ROWS*COLS, n). Backward: w = M @ y_flat.
#
# T-spin policy indices (15-26) get the same geometric A as their no-spin
# counterparts (the spin label has no effect on which cells the piece occupies).

A_matrices = {}
M_matrices = {}

for policy_index in policy_piece_grids_no_padding:
    piece, rotation, t_spin_index = policy_index_to_piece[policy_index]
    # T-spin variants (policy_index 19-26) share geometry with the no-spin
    # rotation (15-18) and reuse the same A/M; skip storing duplicates.
    if t_spin_index != 0:
        continue

    piece_matrix = policy_piece_grids_no_padding[policy_index]
    piece_matrix_rows, piece_matrix_cols = np.shape(piece_matrix)

    kernel_rows = ROWS - piece_matrix_rows + 1
    kernel_cols = COLS - piece_matrix_cols + 1
    kernels = kernel_rows * kernel_cols

    A = np.zeros((ROWS, COLS, kernels), dtype=np.float32)

    for kernel_row in range(kernel_rows):
        for kernel_col in range(kernel_cols):
            kernel_num = kernel_row * kernel_cols + kernel_col
            for piece_row in range(piece_matrix_rows):
                for piece_col in range(piece_matrix_cols):
                    if piece_matrix[piece_row, piece_col] != 0:
                        A[kernel_row + piece_row, kernel_col + piece_col, kernel_num] = 1.0

    A_flat = A.reshape(ROWS * COLS, kernels)
    M_flat = np.linalg.pinv(A_flat).astype(np.float32)

    A_matrices[policy_index] = A
    M_matrices[policy_index] = M_flat.reshape(kernels, ROWS, COLS)



# __main__ stuff

if __name__ == "__main__":
    print(policy_row_range)
    print(policy_col_range)