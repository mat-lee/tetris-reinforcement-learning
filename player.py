from board import Board
from piece_queue import Queue
from stats import Stats
from piece import Piece
from const import *

from numba import

import random

class Player:
    """Parent class for both human and AI players."""
    def __init__(self, ruleset) -> None:
        self.board = Board()
        self.queue = Queue()
        self.stats = Stats(ruleset)

        self.game_over = False
        self.garbage_to_receive = [] # index 0 spawns first
        self.garbage_to_send = [] # sends and receive are in same order

        self.color = None # Used for turn order

        self.piece = None
        self.held_piece = None

        self.draw_coords = None

        self.ruleset = ruleset # Specifies tetr.io s1 or s2 spins

    # Game methods    
    def create_next_piece(self):
        if len(self.queue.pieces) > 0:
            next_piece = self.queue.pieces.pop(0)
            self.create_piece(next_piece)
        elif self.held_piece != None:
            next_piece, self.held_piece = self.held_piece, None
            self.create_piece(next_piece)

    def create_piece(self, piece_type):   
        piece = Piece(piece_dict[piece_type], type=piece_type)
        piece.move_to_spawn()
        if not self.collision(piece.coordinates):
            self.piece = piece
        else:
            # If the block can't spawn lose
            self.game_over = True

    @njit
    def collision(self, coords):    
        """Optimized collision detection using NumPy"""
        # Extract columns and rows for vectorized operations
        cols, rows = coords[:, 0], coords[:, 1]
        
        # Check bounds using vectorized comparison
        out_of_bounds = np.any((rows < 0) | (rows >= ROWS) | (cols < 0) | (cols >= COLS))
        if out_of_bounds:
            return True
        
        # Check board collisions using advanced indexing
        return np.any(self.board.grid[rows, cols] != 0)

    @property
    def ghost_y(self):
        ghost_y = self.piece.location.y
        coordinate_list = self.piece.get_mino_coords(self.piece.location.x, 
                                            ghost_y, 
                                            self.piece.location.rotation, 
                                            self.piece.type)
        
        collided = False
        while collided == False:
            ghost_y += 1
            coordinate_list = coordinate_list + np.array([0, 1], dtype=np.int8)

            if self.collision(coordinate_list):
                collided = True
            
        return ghost_y - 1

    def can_move(self, piece, x_offset=0, y_offset=0):
        """Vectorized movement validation"""
        # Calculate new positions
        new_coords = piece.coordinates + np.array([x_offset, y_offset])
        
        # Check bounds
        cols, rows = new_coords[:, 0], new_coords[:, 1]
        if np.any((rows < 0) | (rows >= ROWS) | (cols < 0) | (cols >= COLS)):
            return False
        
        # Check board collisions
        return not np.any(self.board.grid[rows, cols] != 0)

    def move_right(self):
        if self.can_move(self.piece, x_offset=1):
            self.piece.move(x_offset=1)

    def move_left(self):
        if self.can_move(self.piece, x_offset=-1):
            self.piece.move(x_offset=-1)

    def move_down(self):
        if self.can_move(self.piece, y_offset=1):
            self.piece.move(y_offset=1)

    def try_wallkick(self, dir) -> None:
        """Try to rotate the piece with wallkicks."""
        piece = self.piece
        initial_rotation = piece.location.rotation
        final_rotation = (initial_rotation + dir) % 4

        if piece.type == "I":
            kicktable = i_wallkicks[initial_rotation][final_rotation]
        else:
            kicktable = wallkicks[initial_rotation][final_rotation]

        rotated_piece_coordinates = piece.get_mino_coords(piece.location.x, piece.location.y, final_rotation, piece.type)
        for kick in kicktable: # kicktable is a numpy array itself
            # KICKS ARE NEGATED IN Y DIR IN CONST so it works just noting this here
            test_coords = rotated_piece_coordinates + kick

            # Vectorized bounds and collision check
            cols, rows = test_coords[:, 0], test_coords[:, 1]
            
            # Check bounds
            if np.any((rows < 0) | (rows >= ROWS) | (cols < 0) | (cols >= COLS)):
                continue
                
            # Check collisions
            if np.any(self.board.grid[rows, cols] != 0):
                continue

            # Valid kick
            piece.location.x += kick[0]
            piece.location.y += kick[1]
            piece.location.rotation = final_rotation
            piece.coordinates = test_coords

            # Additionally, check if the piece is a T and the last kick in the kicktable was used:
            if piece.type == "T":
                piece.location.rotation_just_occurred = True
                piece.location.rotation_just_occurred_and_used_last_tspin_kick = False

                if np.array_equal(kick, kicktable[-1]) and dir != 2:
                    piece.location.rotation_just_occurred_and_used_last_tspin_kick = True
            return True
        return False

    def force_place_piece(self, piece_location):
        self.piece.location = piece_location
        self.piece.coordinates = self.piece.get_self_coords
        return self.place_piece()

    def place_piece(self) -> tuple:
        piece = self.piece
        grid = self.board.grid
        stats = self.stats

        place_y = self.ghost_y
        # If the piece was placed in the air, it does not count as a spin
        if piece.location.y != place_y:
            piece.location.rotation_just_occurred = False
            piece.location.rotation_just_occurred_and_used_last_tspin_kick = False

        rows = []
        cleared_rows = []

        is_tspin = False
        is_mini = False
        is_all_clear = False

        # Spins
        # Check for a t-spin
        if piece.type == "T" and piece.location.rotation_just_occurred:
            corners = np.array([[0, 0], [2, 0], [2, 2], [0,  2]]) + np.array([piece.location.x, place_y])
            corner_filled = 4 * [False]

            for i in range(4):
                row = corners[i][1]
                col = corners[i][0]
                if row < 0 or row > ROWS - 1 or col < 0 or col > COLS - 1:
                    corner_filled[i] = True
                elif grid[row][col] != 0:
                    corner_filled[i] = True

            if sum(corner_filled) >= 3:
                is_tspin = True
            
            # Two corner rule + exception
            if not (corner_filled[piece.location.rotation] and corner_filled[(piece.location.rotation + 1) % 4]) and not piece.location.rotation_just_occurred_and_used_last_tspin_kick:
                is_mini = True

        # Check for s2 all spins:
        # Since tetr.io version something the allspin rules apply to t-pieces too
        # If a piece is already a t-spin, don't make it a mini

        ### UPDATE COORDS BEFORE CANMOVE
        self.piece.coordinates = self.piece.get_self_coords

        if self.ruleset == 's2' and not is_tspin:
            # If a piece can't move in any direction it is a mini
            offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
            for x_offset, y_offset in offsets:
                if self.can_move(piece, x_offset=x_offset, y_offset=y_offset):
                    break
            else:
                is_mini = True

        # Place the pieces and check rows that minos will be placed
        final_coords = piece.get_mino_coords(piece.location.x, place_y, piece.location.rotation, piece.type)
        piece_value = piece_type_to_number[piece.type]
        grid[final_coords[:, 1], final_coords[:, 0]] = piece_value
        rows = np.unique(final_coords[:, 1])
        
        self.piece = None

        # Check which rows should be cleared
        for row in rows:
            if np.all(grid[row] != 0): # careful with ghost type
                cleared_rows.append(row)
        
        rows_cleared = len(cleared_rows)

        # If there are any rows cleared, sort the list so they don't overlap
        if rows_cleared > 0:
            cleared_rows.sort()

        # Move rows down one and make a empty line at the top
        for cleared_row in cleared_rows:
            grid[1:cleared_row+1] = grid[0:cleared_row]
            self.board.empty_line(0)

        if rows_cleared > 0:
            # Check if every cell from row `rows_cleared` down is 0
            # Careful with ghost type
            is_all_clear = np.all(grid[rows_cleared:] == 0)

        attack = stats.get_attack(rows_cleared, is_tspin, is_mini, is_all_clear, piece.type) # also updates stats
        stats.pieces += 1

        if attack > 0:
            column = random.randint(0, 9)
            self.garbage_to_send.extend([column] * attack)

        # Help the AI figure out which move the player made
        return rows_cleared
        
    def hold_piece(self):
        if self.held_piece == None:
            self.held_piece = self.piece.type
            self.piece = None
            self.create_next_piece()
        else:
            temp = self.held_piece
            if self.piece == None:
                self.held_piece = None
            else:
                self.held_piece = self.piece.type
            self.create_piece(temp)
    
    def spawn_garbage(self):
        # In MCTS, don't spawn garbage but store it in a variable
        self.board.create_garbage(self.garbage_to_receive)
        self.garbage_to_receive = []
    
    def reset(self):
        self.board = Board()
        self.queue = Queue()
        self.stats = Stats(self.ruleset)
        self.piece = None
        self.held_piece = None
        self.garbage_to_receive = []
        self.game_over = False

    # AI methods
    def copy(self):
        new_player = Player(self.ruleset)
        
        new_player.board = self.board.copy()
        new_player.queue = self.queue.copy()
        new_player.stats.pieces = self.stats.pieces
        new_player.stats.b2b = self.stats.b2b
        new_player.stats.b2b_level = self.stats.b2b_level
        new_player.stats.combo = self.stats.combo
        new_player.game_over = self.game_over
        if self.piece != None:
            new_player.piece = self.piece.copy()
        new_player.held_piece = self.held_piece
        new_player.garbage_to_receive = self.garbage_to_receive[:]
        new_player.color = self.color
        new_player.draw_coords = self.draw_coords

        return new_player

    # Draw methods
    def draw_piece(self, surface, x_0, y_0, rotation, type, color):
        """Draw a piece matrix"""
        coords = Piece.get_mino_coords(x_0, y_0, rotation, type)
        for col, row in coords:
            self.draw_mino(surface, col, row, color)

    def draw_mino(self, surface, col, row, color):
        """Draw a mino"""
        rect = (col * MINO_SIZE + HOLD_WIDTH + E_BUFFER + self.draw_coords[0], 
                    row * MINO_SIZE + N_BUFFER + self.draw_coords[1], MINO_SIZE, MINO_SIZE)
        pygame.draw.rect(surface, color, rect)

    # Show methods
    def show(self, screen):
        """Contains all player's visuals."""
        self.show_ghost(screen)
        self.show_grid_lines(screen)
        self.show_piece(screen)
        self.show_minos(screen)
        self.show_queue(screen)
        self.show_hold(screen)
        self.show_garbage(screen)
        self.show_stats(screen)

    def show_ghost(self, surface):
        if self.piece != None:
            piece = self.piece

            ghost_y = self.ghost_y
            self.draw_piece(surface, piece.location.x, ghost_y, piece.location.rotation, piece.type, color_dict[2]) # 2 corresponds to ghost

    def show_grid_lines(self, surface):
        for row in range(ROWS - GRID_ROWS, ROWS + 1):
            pygame.draw.line(surface, (32, 32, 32), 
                             (HOLD_WIDTH + E_BUFFER + self.draw_coords[0], row * MINO_SIZE + N_BUFFER + self.draw_coords[1]),
                             (HOLD_WIDTH + MINO_SIZE * COLS + E_BUFFER + self.draw_coords[0], row * MINO_SIZE + N_BUFFER + self.draw_coords[1]))
        
        for col in range(COLS + 1):
            pygame.draw.line(surface, (32, 32, 32), 
                             (col * MINO_SIZE + HOLD_WIDTH + E_BUFFER + self.draw_coords[0], MINO_SIZE * (ROWS - GRID_ROWS) + N_BUFFER + self.draw_coords[1]),
                             (col * MINO_SIZE + HOLD_WIDTH + E_BUFFER + self.draw_coords[0], MINO_SIZE * (ROWS) + N_BUFFER + self.draw_coords[1]))

    def show_piece(self, surface):
        if self.piece != None:
            piece = self.piece

            for col, row in piece.coordinates:
                self.draw_mino(surface, col, row, color_dict[piece_type_to_number[piece.type]])
    
    def show_minos(self, surface):
        for row in range(ROWS):
            for col in range(COLS):
                mino = self.board.grid[row][col]
                if mino != 0:
                    self.draw_mino(surface, col, row, color_dict[mino])

    def show_queue(self, surface):
        pieces = self.queue.pieces
        for i in range(PREVIEWS):
            if len(pieces) > i:
                piece = pieces[i]

                x_0 = (12 if piece == "O" else 11) - 0.5
                y_0 = (0 if piece == "I" else 1) + ROWS - SPAWN_ROW
                
                coords = Piece.get_mino_coords(x_0, y_0 + i * 3, 0, piece)
                for col, row in coords:
                    self.draw_mino(surface, col, row, color_dict[piece_type_to_number[piece]])
                            
    def show_hold(self, surface):
        held_piece = self.held_piece
        if held_piece != None:
            x_0 = (-4 if held_piece == "O" else -5) + 0.5
            y_0 = 1 + ROWS - SPAWN_ROW

            coords = Piece.get_mino_coords(x_0, y_0, 0, held_piece)
            for col, row in coords:
                self.draw_mino(surface, col, row, color_dict[piece_type_to_number[held_piece]])

    def show_garbage(self, screen):
        # Show amount of incoming garbage
        # Can't do pygame rects from the bottom
        top_left_x = HOLD_WIDTH + E_BUFFER + self.draw_coords[0] - 1/2 * MINO_SIZE
        top_left_y = ROWS * MINO_SIZE + N_BUFFER + self.draw_coords[1]
        width = MINO_SIZE / 3
        height = len(self.garbage_to_receive) * MINO_SIZE
        
        rect = (top_left_x, top_left_y - height, 
                    width, height)

        pygame.draw.rect(screen, (255, 0, 0), rect)

    def show_stats(self, screen):
        for stat in STAT_SETTINGS:
            text = stat['text']
            if stat['text'] == 'B2B':
                if self.stats.b2b > 0:
                    text = f'B2B X{self.stats.b2b}'
                else:
                    text = None
            elif stat['text'] == 'COMBO':
                if self.stats.combo > 1: 
                    text = f'{self.stats.combo - 1} COMBO'
                else:
                    text = None
            elif stat['text'] == 'attack_text':
                if self.stats.attack_text == '':
                    text = None
                else:
                    text = self.stats.attack_text
            elif stat['text'] == 'pieces_stat': 
                text = str(self.stats.pieces)
            elif stat['text'] == 'attack_stat': 
                text = str(self.stats.lines_sent)

            if stat['text'] == 'LOSES' and self.game_over == False:
                text = None
            
            if text != None:
                font = pygame.font.Font('freesansbold.ttf', stat['size'])
                render_text = font.render(text, True, (255, 255, 255))
                screen.blit(render_text, 
                            (self.draw_coords[0] + E_BUFFER + stat['location'][0], 
                            self.draw_coords[1] + N_BUFFER + (ROWS - GRID_ROWS) * MINO_SIZE + stat['location'][1]))

class Human(Player):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.draw_coords = (0, 0)
        self.color = 0

class AI(Player):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.draw_coords = (WIDTH/2, 0)
        self.color = 1

'''dic = {}
for piece_type in "ZLOSIJT":
    dic[piece_type] = {}
    for r in range(4):
        piece = Piece(piece_dict[piece_type], type=piece_type)
        piece.x_0 = 0
        piece.y_0 = 0
        piece.rotation = r
        piece.update_rotation()
        dic[piece_type][r] = piece.coords
print(dic)'''