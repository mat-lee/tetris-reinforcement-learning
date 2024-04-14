from board import Board
from piece_queue import Queue
from stats import Stats
from piece import Piece
from const import *

import random
import copy

class Player:
    """Parent class for both human and AI players."""
    def __init__(self) -> None:
        self.board = Board()
        self.queue = Queue()
        self.stats = Stats()

        self.game_over = False
        self.garbage_to_receive = [] # index 0 spawns first
        self.garbage_to_send = [] # sends and receive are in same order

        self.piece = None
        self.held_piece = None

        self.draw_coords = None

    # Game methods    
    def create_next_piece(self):
        if len(self.queue.pieces) > 0:
            next_piece = self.queue.pieces.pop(0)
            self.create_piece(next_piece)

    def create_piece(self, piece_type):   
        piece = Piece(piece_dict[piece_type], type=piece_type)
        piece.move_to_spawn()
        if not self.collision(piece.x_0, piece.y_0, piece.matrix):
            self.piece = piece
        else:
            # If the block can't spawn lose
            self.game_over = True

    def collision(self, x_0, y_0, matrix):
        grid = self.board.grid

        coords = Piece.get_mino_coords(x_0, y_0, matrix)

        for col, row in coords:
            if row < 0 or row > ROWS - 1 or col < 0 or col > COLS - 1:
                return True
            elif grid[row][col].type != "empty":
                return True

        return False

    @property
    def ghost_y(self):
        ghost_y = self.piece.y_0
        while(not self.collision(self.piece.x_0, ghost_y + 1, self.piece.matrix)):
            ghost_y += 1
        
        return(ghost_y)

    def can_move(self, x_offset=0, y_offset=0):
        if not self.collision(self.piece.x_0 + x_offset, 
                              self.piece.y_0 + y_offset, self.piece.matrix):
            return True
        return False

    def move_right(self):
        if self.can_move(x_offset=1):
            self.piece.x_0 += 1

    def move_left(self):
        if self.can_move(x_offset=-1):
            self.piece.x_0 -= 1

    def move_down(self):
        if self.can_move(y_offset=1):
            self.piece.y_0 += 1

    def try_wallkick(self, dir):
        piece = self.piece
        i_rotation = piece.rotation
        f_rotation = (i_rotation + dir) % 4

        key = str(i_rotation) + "-" + str(f_rotation)

        if piece.type == "I":
            kicktable = i_wallkicks[key]
        else:
            kicktable = wallkicks[key]

        rotated_piece_matrix = piece.rotated_matrix(piece.rotation, f_rotation)

        for kick in kicktable:
            if not self.collision(piece.x_0 + kick[0], piece.y_0 - kick[1], rotated_piece_matrix):
                piece.x_0 += kick[0]
                piece.y_0 += -kick[1]
                piece.matrix = rotated_piece_matrix
                piece.rotation = f_rotation
                return()

    def force_place_piece(self, x, y, o):
        self.piece.x_0 = x
        self.piece.y_0 = y
        self.piece.rotation = o
        self.piece.update_rotation()
        self.place_piece()

    def place_piece(self):
        piece = self.piece
        grid = self.board.grid
        stats = self.stats

        place_y = self.ghost_y

        rows = []
        cleared_rows = []

        is_tspin = False
        is_mini = False
        is_all_clear = False

        # Check for a t-spin
        if piece.type == "T":
            corners = [[0, 0], [2, 0], [2, 2], [0,  2]]
            corner_filled = 4 * [False]

            for i in range(4):
                row = corners[i][1] + place_y
                col = corners[i][0] + piece.x_0 
                if row < 0 or row > ROWS - 1 or col < 0 or col > COLS - 1:
                    corner_filled[i] = True
                elif grid[row][col].type != "empty":
                    corner_filled[i] = True

            if sum(corner_filled) >= 3:
                is_tspin = True
            
            if not (corner_filled[piece.rotation] and corner_filled[(piece.rotation + 1) % 4]):
                is_mini = True
            
        # Check the rows the minos will be placed in
        for y in range(len(piece.matrix)):
            if any(x != 0 for x in piece.matrix[y]):
                rows.append(int(place_y + y))

        # Place the pieces
        coords = Piece.get_mino_coords(piece.x_0, place_y, piece.matrix)

        print("simul/place", piece.x_0, piece.y_0, piece.rotation)
        for col, row in coords:
            grid[row][col].type = piece.type
        
        self.piece = None

        # Check which rows should be cleared
        for row in rows:
            if all(mino.type != "empty" for mino in grid[row]): # careful with ghost type
                cleared_rows.append(row)
        
        rows_cleared = len(cleared_rows)

        # Move rows down one and make a empty line at the top
        for cleared_row in cleared_rows:
            for row in range(cleared_row)[::-1]:
                for col in range(COLS):
                    grid[row + 1][col].type = grid[row][col].type

            self.board.empty_line(0)

        if rows_cleared > 0:
            is_all_clear = True
            for row in range(rows_cleared, ROWS): # rows_cleared num of empty lines at top
                for col in range(COLS):
                    if grid[row][col].type != "empty": # careful with ghost type
                        is_all_clear = False
                        break

        attack = stats.get_attack(rows_cleared, is_tspin, is_mini, is_all_clear) # also updates stats
        stats.pieces += 1

        if attack > 0:
            column = random.randint(0, 9)
            self.garbage_to_send.extend([column] * attack)

        # Help the AI figure out which move the player made
        location_placed = (piece.x_0, place_y, piece.rotation)
        return location_placed
        
    def hold_piece(self):
        if self.held_piece == None:
            self.held_piece = self.piece.type
            self.piece = None
            self.create_next_piece()
        else:
            temp = self.held_piece
            self.held_piece = self.piece.type
            self.create_piece(temp)
    
    def spawn_garbage(self):
        self.board.create_garbage(self.garbage_to_receive)
        self.garbage_to_receive = []
    
    def reset(self):
        self.board = Board()
        self.queue = Queue()
        self.stats = Stats()
        self.piece = None
        self.held_piece = None
        self.garbage_to_receive = []
        self.game_over = False

    # Draw methods
    def draw_piece(self, surface, x_0, y_0, piece_matrix, color):
        """Draw a piece matrix"""
        coords = Piece.get_mino_coords(x_0, y_0, piece_matrix)
        for coord in coords:
            col, row = coord
            self.draw_mino(surface, row, col, color)

    def draw_mino(self, surface, row, col, color):
        """Draw a mino"""
        rect = (col * MINO_SIZE + HOLD_WIDTH + E_BUFFER + self.draw_coords[0], 
                    row * MINO_SIZE + N_BUFFER + self.draw_coords[1], MINO_SIZE, MINO_SIZE)
        pygame.draw.rect(surface, color, rect)

    # Show methods
    def show_player(self, screen):
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
            self.draw_piece(surface, piece.x_0, ghost_y, piece.matrix, color_dict["ghost"])

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

            self.draw_piece(surface, piece.x_0, piece.y_0, piece.matrix, color_dict[piece.type])
    
    def show_minos(self, surface):
        for row in range(ROWS):
            for col in range(COLS):
                mino = self.board.grid[row][col]
                if mino.type != "empty":
                    self.draw_mino(surface, mino.row, mino.col, color_dict[mino.type])

    def show_queue(self, surface):
        pieces = self.queue.pieces
        for i in range(PREVIEWS):
            if len(pieces) > i:
                piece = pieces[i]
                piece_matrix = piece_dict[piece]

                x_0 = (12 if piece == "O" else 11) - 0.5
                y_0 = (0 if piece == "I" else 1) + ROWS - SPAWN_ROW
                
                coords = Piece.get_mino_coords(x_0, y_0 + i * 3, piece_matrix)
                for coord in coords:
                    col, row = coord
                    self.draw_mino(surface, row, col, color_dict[piece])
                            
    def show_hold(self, surface):
        held_piece = self.held_piece
        if held_piece != None:
            piece_matrix = piece_dict[held_piece]

            x_0 = (-4 if held_piece == "O" else -5) + 0.5
            y_0 = 1 + ROWS - SPAWN_ROW

            coords = Piece.get_mino_coords(x_0, y_0, piece_matrix)
            for coord in coords:
                col, row = coord
                self.draw_mino(surface, row, col, color_dict[held_piece])

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
            text = copy.deepcopy(stat['text'])
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
                            self.draw_coords[1] + N_BUFFER + stat['location'][1]))

class Human(Player):
    def __init__(self) -> None:
        super().__init__()
        self.draw_coords = (0, 0)