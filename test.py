import pygame
import random

# Initialize Pygame
pygame.init()

# Define constants
GRID_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SCREEN_WIDTH = GRID_WIDTH * GRID_SIZE * 2
SCREEN_HEIGHT = GRID_HEIGHT * GRID_SIZE
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Define tetromino shapes and colors
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1, 0], [0, 1, 1]],  # Z
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[0, 1, 0], [1, 1, 1]],  # L
    [[1, 1, 1], [1, 0, 0]]  # J
]

COLORS = [CYAN, RED, GREEN, YELLOW, PURPLE, ORANGE, BLUE]

# Define the Tetris board class
class TetrisBoard:
    def __init__(self, x_offset, ai=False):
        self.x_offset = x_offset
        self.grid = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.current_shape = self.create_random_tetromino()
        self.ai = ai

    def create_random_tetromino(self):
        return Tetromino(3 * GRID_SIZE, 0, random.choice(SHAPES), random.choice(COLORS))

    def update(self, keys):
        self.current_shape.update(keys, self.grid)
        if not self.current_shape.active:
            self.current_shape = self.create_random_tetromino()

    def render(self, screen):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                color = self.grid[y][x]
                if color:
                    pygame.draw.rect(screen, color, (self.x_offset + x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        self.current_shape.render(screen, self.x_offset)

    def clear_lines(self):
        lines_cleared = 0
        for y in range(GRID_HEIGHT - 1, -1, -1):
            if all(self.grid[y]):
                lines_cleared += 1
                del self.grid[y]
                self.grid.insert(0, [0] * GRID_WIDTH)
        return lines_cleared

# Define the Tetromino class
class Tetromino:
    def __init__(self, x, y, shape, color):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = color
        self.active = True
        self.last_move_time = pygame.time.get_ticks()
        self.fall_speed = 500

    def update(self, keys, grid):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_move_time > self.fall_speed:
            self.move(0, GRID_SIZE)
            if self.collision(grid):
                self.move(0, -GRID_SIZE)
                self.lock(grid)
                self.active = False
            self.last_move_time = current_time

        self.handle_input(keys, grid)

    def handle_input(self, keys, grid):
        if keys[pygame.K_LEFT]:
            self.move(-GRID_SIZE, 0)
            if self.collision(grid):
                self.move(GRID_SIZE, 0)
        elif keys[pygame.K_RIGHT]:
            self.move(GRID_SIZE, 0)
            if self.collision(grid):
                self.move(-GRID_SIZE, 0)
        elif keys[pygame.K_DOWN]:
            self.move(0, GRID_SIZE)
            if self.collision(grid):
                self.move(0, -GRID_SIZE)
        elif keys[pygame.K_UP]:
            self.rotate(grid)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def rotate(self, grid):
        original_shape = self.shape
        self.shape = list(zip(*self.shape[::-1]))
        if self.collision(grid):
            self.shape = original_shape

    def collision(self, grid):
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell and (self.x / GRID_SIZE + x < 0 or self.x / GRID_SIZE + x >= GRID_WIDTH or
                            self.y / GRID_SIZE + y >= GRID_HEIGHT or grid[int(self.y / GRID_SIZE) + y][
                                int(self.x / GRID_SIZE) + x]):
                    return True
        return False

    def lock(self, grid):
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid[int(self.y / GRID_SIZE) + y][int(self.x / GRID_SIZE) + x] = self.color

    def render(self, screen, x_offset):
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, self.color, (x_offset + self.x + x * GRID_SIZE, self.y + y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

# Function to send lines from one board to another
def send_lines(sender, receiver, lines):
    for _ in range(lines):
        del receiver.grid[0]
        receiver.grid.append([sender.color] * GRID_WIDTH)

# Function to run the game
def run_game():
    player_board = TetrisBoard(0)
    ai_board = TetrisBoard(SCREEN_WIDTH // 2, ai=True)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Update both player and AI boards
        player_board.update(keys)
        ai_board.update(keys)

        # Clear lines and send lines
        lines_cleared = player_board.clear_lines()
        if lines_cleared > 0:
            send_lines(player_board, ai_board, lines_cleared)

        lines_cleared = ai_board.clear_lines()
        if lines_cleared > 0:
            send_lines(ai_board, player_board, lines_cleared)

        # Render the screen
        SCREEN.fill(BLACK)
        player_board.render(SCREEN)
        ai_board.render(SCREEN)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == '__main__':
    run_game()
