from board import Board
from piece_queue import Queue
from stats import Stats
from piece import Piece
from const import *
import random

class Player:
    """Parent class for both human and AI players."""
    def __init__(self) -> None:
        self.board = Board()
        self.queue = Queue()
        self.stats = Stats()

        self.game_over = False
        self.garbage_to_receive = [] # index 0 appears first
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

    def get_mino_coords(self, _x_0, _y_0, piece_matrix):
        coordinate_list = []
        cols = len(piece_matrix[0])
        rows = len(piece_matrix)
        for x in range(cols):
            for y in range(rows):
                if piece_matrix[y][x] != 0:
                    col = _x_0 + x
                    row = _y_0 + y
                    coordinate_list.append([col, row])
        
        return(coordinate_list)

    def collision(self, _x_0, _y_0, matrix):
        grid = self.board.grid

        coords = self.get_mino_coords(_x_0, _y_0, matrix)

        for coord in coords:
            col, row = coord
            if row < 0 or row > ROWS - 1 or col < 0 or col > COLS - 1:
                return True
            elif grid[row][col].type != "empty":
                return True

        return False

    def get_ghost_y(self):
        piece = self.piece

        ghost_y = piece.y_0
        while(not self.collision(piece.x_0, ghost_y + 1, piece.matrix)):
            ghost_y += 1
        
        return(ghost_y)

    def can_move_right(self):
        piece = self.piece

        if not self.collision(piece.x_0 + 1, piece.y_0, piece.matrix):
            return True
        return False

    def move_right(self):
        if self.can_move_right():
            self.piece.x_0 += 1

    def can_move_left(self):
        piece = self.piece

        if not self.collision(piece.x_0 - 1, piece.y_0, piece.matrix):
            return True
        return False

    def move_left(self):
        if self.can_move_left():
            self.piece.x_0 -= 1
    
    def can_move_down(self):
        piece = self.piece

        if not self.collision(piece.x_0 , piece.y_0 + 1, piece.matrix):
            return True
        return False

    def move_down(self):
        if self.can_move_down():
            self.piece.y_0 += 1
    
    def rotate_piece(self, piece, final):
        diff = (final - piece.rotation) % 4

        # Clockwise rotation
        if diff == 1:
            return(list(zip(*piece.matrix[::-1])))

        # 180 rotation
        if diff == 2:
            return([x[::-1] for x in piece.matrix[::-1]])

        # Counter-clockwise rotation
        if diff == 3:
            return(list(zip(*piece.matrix))[::-1])

    def try_wallkick(self, dir):
        piece = self.piece
        i_rotation = piece.rotation
        f_rotation = (i_rotation + dir) % 4

        key = str(i_rotation) + "-" + str(f_rotation)

        if piece.type == "I":
            kicktable = i_wallkicks[key]
        else:
            kicktable = wallkicks[key]

        rotated_piece_matrix = self.rotate_piece(piece, f_rotation)

        for kick in kicktable:
            if not self.collision(piece.x_0 + kick[0], piece.y_0 - kick[1], rotated_piece_matrix):
                piece.x_0 += kick[0]
                piece.y_0 += -kick[1]
                piece.matrix = rotated_piece_matrix
                piece.rotation = f_rotation
                return()

    def place_piece(self):
        piece = self.piece
        grid = self.board.grid
        stats = self.stats

        place_y = self.get_ghost_y()

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
                else:
                    if grid[row][col].type != "empty":
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
        coords = self.get_mino_coords(piece.x_0, place_y, piece.matrix)

        for coord in coords:
            col, row = coord
            grid[row][col].type = piece.type
        
        self.piece = None

        # Check which rows should be cleared
        for row in rows:
            if all(mino.type != "empty" for mino in grid[row]):
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
            for row in range(ROWS):
                for col in range(COLS):
                    if grid[row][col].type != "empty":
                        is_all_clear = False
                        break

        attack = stats.get_attack(rows_cleared, is_tspin, is_mini, is_all_clear) # also updates stats
        stats.pieces += 1

        if attack > 0:
            column = random.randint(0, 9)
            for _ in range(attack):
                self.garbage_to_send.append(column)
        
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
        for column in self.garbage_to_receive: # Using pop changes the order of the list
            self.board.garbage_line(column)
        
        self.garbage_to_receive = []

    # Draw methods
    def draw_piece(self, surface, _x_0, _y_0, piece_matrix, color):
        """Draw a piece matrix"""
        coords = self.get_mino_coords(_x_0, _y_0, piece_matrix)
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

            ghost_y = self.get_ghost_y()
            self.draw_piece(surface, piece.x_0, ghost_y, piece.matrix, color_dict["ghost"])

    def show_grid_lines(self, surface):
        for row in range(ROWS + 1):
            pygame.draw.line(surface, (32, 32, 32), 
                             (HOLD_WIDTH + E_BUFFER + self.draw_coords[0], row * MINO_SIZE + N_BUFFER + self.draw_coords[1]),
                             (HOLD_WIDTH + MINO_SIZE * COLS + E_BUFFER + self.draw_coords[0], row * MINO_SIZE + N_BUFFER + self.draw_coords[1]))
        
        for col in range(COLS + 1):
            pygame.draw.line(surface, (32, 32, 32), 
                             (col * MINO_SIZE + HOLD_WIDTH + E_BUFFER + self.draw_coords[0], N_BUFFER + self.draw_coords[1]),
                             (col * MINO_SIZE + HOLD_WIDTH + E_BUFFER + self.draw_coords[0], MINO_SIZE * ROWS + N_BUFFER + self.draw_coords[1]))

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
                y_0 = (0 if piece == "I" else 1)
                
                coords = self.get_mino_coords(x_0, y_0 + i * 3, piece_matrix)
                for coord in coords:
                    col, row = coord
                    self.draw_mino(surface, row, col, color_dict[piece])
                            
    def show_hold(self, surface):
        held_piece = self.held_piece
        if held_piece != None:
            piece_matrix = piece_dict[held_piece]

            x_0 = (-4 if held_piece == "O" else -5) + 0.5
            y_0 = 1

            coords = self.get_mino_coords(x_0, y_0, piece_matrix)
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
        factor = MINO_SIZE / 30
        font = pygame.font.Font('freesansbold.ttf', int(18 * factor))

        render_lists = [["PIECES", self.stats.pieces, 5],
                       ["ATTACK", self.stats.lines_sent, 6],
                       ["B2B", self.stats.b2b, 7],
                       ["COMBO", self.stats.combo, 8]]
        
        for render_list in render_lists:
            text = font.render(f'{render_list[0]}: {render_list[1]}', True, (255, 255, 255))
            screen.blit(text, (1 * MINO_SIZE + E_BUFFER + self.draw_coords[0], 
                           render_list[2] * MINO_SIZE + N_BUFFER + self.draw_coords[1]))
        
        if self.game_over == True:
            text = font.render('LOSES', True, (255, 255, 255))
            screen.blit(text, (1 * MINO_SIZE + E_BUFFER + self.draw_coords[0],
                           9 * MINO_SIZE + N_BUFFER + self.draw_coords[1]))

        '''text_5 = font.render(f'LEVEL: {self.stats.b2b_level}', True, (255, 255, 255))
        screen.blit(text_5, (1 * MINO_SIZE + E_BUFFER, 
                           9 * MINO_SIZE + N_BUFFER))'''

class Human(Player):
    def __init__(self) -> None:
        super().__init__()
        self.draw_coords = (0, 0)

class AI(Player):
    def __init__(self) -> None:
        super().__init__()
        self.draw_coords = (WIDTH/2, 0)
    
    def make_move(self):
        self.place_piece()