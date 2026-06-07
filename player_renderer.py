from const import *
from piece import Piece


class PlayerRenderer:
    """Handles all pygame drawing for a single Player. Keeps rendering out of game logic."""

    def __init__(self, player, draw_coords):
        self.player = player
        self.draw_coords = draw_coords
        self.show_value_estimate = False

    def show(self, screen):
        self.show_ghost(screen)
        self.show_grid_lines(screen)
        self.show_piece(screen)
        self.show_minos(screen)
        self.show_queue(screen)
        self.show_hold(screen)
        self.show_garbage(screen)
        self.show_stats(screen)

    def draw_mino(self, surface, col, row, color):
        rect = (col * MINO_SIZE + HOLD_WIDTH + E_BUFFER + self.draw_coords[0],
                row * MINO_SIZE + N_BUFFER + self.draw_coords[1], MINO_SIZE, MINO_SIZE)
        pygame.draw.rect(surface, color, rect)

    def draw_piece(self, surface, x_0, y_0, rotation, piece_type, color):
        for col, row in Piece.get_mino_coords(x_0, y_0, rotation, piece_type):
            self.draw_mino(surface, col, row, color)

    def show_ghost(self, surface):
        player = self.player
        if player.piece is not None:
            piece = player.piece
            self.draw_piece(surface, piece.location.x, player.ghost_y,
                            piece.location.rotation, piece.type, color_dict["ghost"])

    def show_grid_lines(self, surface):
        dx, dy = self.draw_coords
        for row in range(ROWS - GRID_ROWS, ROWS + 1):
            pygame.draw.line(surface, (32, 32, 32),
                             (HOLD_WIDTH + E_BUFFER + dx, row * MINO_SIZE + N_BUFFER + dy),
                             (HOLD_WIDTH + MINO_SIZE * COLS + E_BUFFER + dx, row * MINO_SIZE + N_BUFFER + dy))
        for col in range(COLS + 1):
            pygame.draw.line(surface, (32, 32, 32),
                             (col * MINO_SIZE + HOLD_WIDTH + E_BUFFER + dx, MINO_SIZE * (ROWS - GRID_ROWS) + N_BUFFER + dy),
                             (col * MINO_SIZE + HOLD_WIDTH + E_BUFFER + dx, MINO_SIZE * ROWS + N_BUFFER + dy))

    def show_piece(self, surface):
        player = self.player
        if player.piece is not None:
            for col, row in player.piece.coordinates:
                self.draw_mino(surface, col, row, color_dict[player.piece.type])

    def show_minos(self, surface):
        grid = self.player.board.grid
        for row in range(ROWS):
            for col in range(COLS):
                mino = grid[row][col]
                if mino != 0:
                    self.draw_mino(surface, col, row, color_dict[mino])

    def show_queue(self, surface):
        for i, piece in enumerate(self.player.queue.pieces[:PREVIEWS]):
            x_0 = (12 if piece == "O" else 11) - 0.5
            y_0 = (0 if piece == "I" else 1) + ROWS - SPAWN_ROW
            for col, row in Piece.get_mino_coords(x_0, y_0 + i * 3, 0, piece):
                self.draw_mino(surface, col, row, color_dict[piece])

    def show_hold(self, surface):
        held = self.player.held_piece
        if held is not None:
            x_0 = (-4 if held == "O" else -5) + 0.5
            y_0 = 1 + ROWS - SPAWN_ROW
            for col, row in Piece.get_mino_coords(x_0, y_0, 0, held):
                self.draw_mino(surface, col, row, color_dict[held])

    def show_garbage(self, screen):
        dx, dy = self.draw_coords
        top_left_x = HOLD_WIDTH + E_BUFFER + dx - 0.5 * MINO_SIZE
        top_left_y = ROWS * MINO_SIZE + N_BUFFER + dy
        height = len(self.player.garbage_to_receive) * MINO_SIZE
        pygame.draw.rect(screen, (255, 0, 0), (top_left_x, top_left_y - height, MINO_SIZE / 3, height))

    def show_stats(self, screen):
        player = self.player
        for stat in STAT_SETTINGS:
            text = stat['text']
            if stat['text'] == 'B2B':
                text = f'B2B X{player.stats.b2b}' if player.stats.b2b > 0 else None
            elif stat['text'] == 'COMBO':
                text = f'{player.stats.combo - 1} COMBO' if player.stats.combo > 1 else None
            elif stat['text'] == 'attack_text':
                text = player.stats.attack_text or None
            elif stat['text'] == 'pieces_stat':
                text = str(player.stats.pieces)
            elif stat['text'] == 'attack_stat':
                text = str(player.stats.lines_sent)
            elif stat['text'] == 'EVAL':
                text = 'EVAL' if self.show_value_estimate else None
            elif stat['text'] == 'value_stat':
                text = (f'{player.stats.value_estimate:.2f}'
                        if self.show_value_estimate and player.stats.value_estimate is not None else None)

            if stat['text'] == 'LOSES' and not player.game_over:
                text = None

            if text is not None:
                font = pygame.font.Font('freesansbold.ttf', stat['size'])
                render_text = font.render(text, True, (255, 255, 255))
                screen.blit(render_text,
                            (self.draw_coords[0] + E_BUFFER + stat['location'][0],
                             self.draw_coords[1] + N_BUFFER + (ROWS - GRID_ROWS) * MINO_SIZE + stat['location'][1]))
