import pygame
from numpy import prod

# Board Dimensions:
ROWS = 26
SPAWN_ROW = 23
GRID_ROWS = 20

GRID_ROWS = 20
COLS = 10

# AI settings
MAX_ITER = 80 # 1600
MAX_MOVES = 1000

TRAINING_GAMES = 200 # 25000
TRAINING_LOOPS = 1 # 1000
SETS_TO_TRAIN_WITH = 20 # 20
BATTLE_GAMES = 200 # 400

# Controls:
k_move_left =   pygame.K_LEFT
k_move_right =  pygame.K_RIGHT
k_soft_drop =   pygame.K_DOWN
k_hard_drop =   pygame.K_SPACE
k_make_ai_move = pygame.K_F
k_rotate_ccw =  pygame.K_z
k_rotate_cw =   pygame.K_UP
k_rotate_180 =  pygame.K_a
k_hold =        pygame.K_c
k_undo =        pygame.K_1
k_redo =        pygame.K_2
k_restart =     pygame.K_r

# graphics break when ARR = 0
# idc
DAS = 100
ARR = 0
SDF = 1/10

PREVIEWS = 5

# Screen Dimensions:
MINO_SIZE = 30

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
    {'text': "attack_stat", 'size': int(MINO_SIZE * 0.9 ), 'location': (HOLD_WIDTH - 1.5 * MINO_SIZE, 18 * MINO_SIZE)}
]

MINOS = "ZLOSIJT"

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
    0: ["O", 0],
    1: ["Z", 0],
    2: ["Z", 1],
    3: ["S", 0],
    4: ["S", 1],
    5: ["I", 0],
    6: ["I", 1],
    7: ["L", 0],
    8: ["L", 1],
    9: ["L", 2],
    10: ["L", 3],
    11: ["J", 0],
    12: ["J", 1],
    13: ["J", 2],
    14: ["J", 3],
    15: ["T", 0],
    16: ["T", 1],
    17: ["T", 2],
    18: ["T", 3],
}

# Leftside buffer is 2
POLICY_SHAPE = (len(policy_index_to_piece), ROWS - 1, COLS + 2 - 1)
POLICY_SIZE = prod(POLICY_SHAPE)

policy_piece_to_index = {
    "O": {0: 0},
    "Z": {0: 1, 1: 2},
    "S": {0: 3, 1: 4},
    "I": {0: 5, 1: 6},
    "L": {0: 7, 1: 8, 2: 9, 3: 10},
    "J": {0: 11, 1: 12, 2: 13, 3: 14},
    "T": {0: 15, 1: 16, 2: 17, 3: 18},
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
        2: [[0, 0]],
    },
    1: {
        0: [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
        2: [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
        3: [[0, 0]],
    },
    2: {
        1: [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
        3: [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
        0: [[0, 0]],
    },
    3: {
        2: [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
        0: [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
        1: [[0, 0]],
    },
}

i_wallkicks = {
    0: {
        1: [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
        3: [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
        2: [[0, 0]],
    },
    1: {
        0: [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
        2: [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
        3: [[0, 0]],
    },
    2: {
        1: [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
        3: [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
        0: [[0, 0]],
    },
    3: {
        2: [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
        0: [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
        1: [[0, 0]],
    },
}
