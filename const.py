import pygame

# Controls:
k_move_left =   pygame.K_LEFT
k_move_right =  pygame.K_RIGHT
k_soft_drop =   pygame.K_DOWN
k_hard_drop =   pygame.K_SPACE
k_rotate_ccw =  pygame.K_z
k_rotate_cw =   pygame.K_UP
k_rotate_180 =  pygame.K_a
k_hold =        pygame.K_c
k_undo =        pygame.K_1
k_redo =        pygame.K_2

# graphics break when ARR = 0
# idc
DAS = 100
ARR = 0
SDF = 1/10

PREVIEWS = 5

# Board Dimensions:
ROWS = 23
GRID_ROWS = 20
COLS = 10

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


# Piece Matrices:
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
    "empty":   (0,   0,   0),
    "ghost":   (32,  32,  32),
    "garbage": (85,  85, 85),
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
  "0-1": [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
  "1-0": [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
  "1-2": [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
  "2-1": [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
  "2-3": [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
  "3-2": [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
  "3-0": [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
  "0-3": [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
  "0-2": [[0, 0]],
  "1-3": [[0, 0]],
  "2-0": [[0, 0]],
  "3-1": [[0, 0]],
}

i_wallkicks = {
  "0-1": [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
  "1-0": [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
  "1-2": [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
  "2-1": [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
  "2-3": [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
  "3-2": [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
  "3-0": [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
  "0-3": [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
  "0-2": [[0, 0]],
  "1-3": [[0, 0]],
  "2-0": [[0, 0]],
  "3-1": [[0, 0]],
};