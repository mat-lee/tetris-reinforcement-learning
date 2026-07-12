from const import *
import numpy as np
from scipy import signal
from collections import deque
from piece_location import PieceLocation

# Binary masks for each (piece type, rotation), precomputed once.
# Built from mino_coords_dict and cropped to the mino bounding box, so that
# convolution results are anchored at the topmost, leftmost mino (policy coords).
def _mino_mask(piece_type, rotation):
    coords = mino_coords_dict[piece_type][rotation]
    col_buffer = piece_rotation_col_buffer[piece_type][rotation]
    row_buffer = piece_rotation_row_buffer[piece_type][rotation]
    height = max(row for col, row in coords) - row_buffer + 1
    width = max(col for col, row in coords) - col_buffer + 1
    mask = np.zeros((height, width), dtype=np.int32)
    for col, row in coords:
        mask[row - row_buffer][col - col_buffer] = 1
    return mask

PIECE_MASKS = {
    piece_type: [_mino_mask(piece_type, rotation) for rotation in range(4)]
    for piece_type in mino_coords_dict
}


class MoveGenerator:
    """Handles piece movement generation with multiple algorithms."""
    
    def __init__(self, player, policy_shape, policy_pieces, policy_piece_to_index, 
                 piece_dict, mino_coords_dict, rows, spawn_row):
        self.player = player
        self.POLICY_SHAPE = policy_shape
        self.policy_pieces = policy_pieces
        self.policy_piece_to_index = policy_piece_to_index
        self.piece_dict = piece_dict
        self.mino_coords_dict = mino_coords_dict
        self.ROWS = rows
        self.SPAWN_ROW = spawn_row
        
        # State for current generation
        self.sim_player = None
        self.piece = None
        self.next_location_queue = None
        self.place_location_queue = None
        self.checked_list = None
        # Per-rotation true coords -> policy coords buffers for the current piece
        self.row_buffer = None
        self.col_buffer = None
        
    def generate_moves(self, algorithm='brute-force'):
        """Main entry point for move generation."""
        new_policy = np.zeros(self.POLICY_SHAPE)

        # Occupancy grid (1 = blocked), shared by both piece types
        self.occupancy = np.array(
            [[0 if cell == 0 else 1 for cell in row] for row in self.player.board.grid],
            dtype=np.int32)
        
        # Try both current piece and held piece
        piece_types = self._get_piece_types_to_check()
        
        for piece_type in piece_types:
            if piece_type is not None:
                piece_moves = self._generate_moves_for_piece(piece_type, algorithm)
                new_policy = np.logical_or(new_policy, piece_moves)
        
        return new_policy
    
    def _get_piece_types_to_check(self):
        """Get the piece types to check (current and held)."""
        sim_player = self.player.copy()
        piece_1 = sim_player.piece.type if sim_player.piece else None
        
        sim_player.hold_piece()
        piece_2 = sim_player.piece.type if sim_player.piece else None
        
        # Only return unique piece types
        if piece_1 == piece_2:
            return [piece_1] if piece_1 is not None else []
        return [p for p in [piece_1, piece_2] if p is not None]
    
    def _generate_moves_for_piece(self, piece_type, algorithm):
        """Generate all possible moves for a specific piece type."""
        policy_matrix = np.zeros(self.POLICY_SHAPE)
        
        # Set up simulation state
        self.sim_player = self.player.copy()
        if self.sim_player.piece is None or self.sim_player.piece.type != piece_type:
            self.sim_player.hold_piece()
        
        self.piece = self.sim_player.piece
        if self.piece is None:
            return policy_matrix

        self.row_buffer = piece_rotation_row_buffer[self.piece.type]
        self.col_buffer = piece_rotation_col_buffer[self.piece.type]

        # Initialize tracking structures
        self._initialize_tracking_structures()
        
        # Set starting position
        self._set_starting_position()
        
        # Generate moves using specified algorithm
        algorithm_map = {
            'brute-force': self._brute_force_algorithm,
            'faster-but-loss': self._optimized_algorithm,
            'harddrop': self._harddrop_algorithm,
            'convolutional': self._convolutional_algorithm,
            # 'conv-optimized': self._conv_optimized_algorithm
        }
        
        if algorithm not in algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Determine if piece can rotate
        check_rotations = self.piece.type != "O"
        
        algorithm_map[algorithm](check_rotations)
        
        # Convert placements to policy matrix
        return self._convert_placements_to_policy()
    
    def _initialize_tracking_structures(self):
        """Initialize queues and tracking arrays."""
        self.next_location_queue = deque()
        self.place_location_queue = []
        
        # The action space in policy format: [rotation, row, col, t_spin_state],
        # where (row, col) are policy coords
        t_spin_states = 3 if self.piece.type == "T" else 1
        self.checked_list = np.zeros((4, ROWS, COLS, t_spin_states), dtype=int)
    
    def _set_starting_position(self):
        """Set the piece to its starting position."""
        highest_row = self._get_highest_row()
        starting_row = max(highest_row - len(self.piece_dict[self.piece.type]), 
                          self.ROWS - self.SPAWN_ROW)
        self.piece.location.y = starting_row
        
        self._check_add_to_sets(self.piece.type, self.piece.location.copy(), check_placement=True)
    
    def _get_highest_row(self):
        """Find the highest occupied row in the grid."""
        grid = self.sim_player.board.grid
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] != 0:
                    return row
        return len(grid)
    
    def _check_add_to_sets(self, piece_type, piece_location, check_placement=False):
        """Add a location to be checked if not already processed."""
        if not self._already_checked(piece_type, piece_location):
            # Special handling for T-piece rotation states
            should_skip = False
            if piece_type == "T" and piece_location.rotation_just_occurred:
                non_rotated_location = piece_location.copy()
                non_rotated_location.rotation_just_occurred = False
                if self._already_checked(piece_type, non_rotated_location):
                    should_skip = True
            
            if not should_skip:
                self.next_location_queue.append(piece_location)
                self._mark_checked(piece_type, piece_location)
            
            # Check if it can be placed
            if check_placement:
                coords = [[col + piece_location.x, row + piece_location.y + 1] 
                         for col, row in self.mino_coords_dict[self.piece.type][piece_location.rotation]]
                if self.sim_player.collision(coords):
                    self.place_location_queue.append(piece_location)
    
    def _checked_index(self, piece_location):
        """Convert a location to its policy-coord index in checked_list."""
        rotation = piece_location.rotation
        index = 0
        if piece_location.rotation_just_occurred:
            index = 1
        if piece_location.rotation_just_occurred_and_used_last_tspin_kick:
            index = 2
        return (rotation,
                piece_location.y + self.row_buffer[rotation],
                piece_location.x + self.col_buffer[rotation],
                index)

    def _already_checked(self, piece_type, piece_location):
        """Check if a location has already been processed."""
        return self.checked_list[self._checked_index(piece_location)] == 1

    def _mark_checked(self, piece_type, piece_location):
        """Mark a location as processed."""
        self.checked_list[self._checked_index(piece_location)] = 1
    
    def _brute_force_algorithm(self, check_rotations):
        """Exhaustive search algorithm that finds all possible moves."""
        while len(self.next_location_queue) > 0:
            piece_location = self.next_location_queue.popleft()
            piece_location_copy = piece_location.copy()
            
            self.piece.location = piece_location_copy.copy()
            self.piece.coordinates = self.piece.get_self_coords
            
            # Check left, right, and down moves
            for move in [[1, 0], [-1, 0], [0, 1]]:
                if self.sim_player.can_move(self.piece, x_offset=move[0], y_offset=move[1]):
                    new_location = self.piece.location.copy()
                    new_location.x += move[0]
                    new_location.y += move[1]
                    new_location.rotation_just_occurred = False
                    new_location.rotation_just_occurred_and_used_last_tspin_kick = False
                    
                    self._check_add_to_sets(self.piece.type, new_location, check_placement=True)
            
            # Check rotations
            if check_rotations:
                for i in range(1, 4):
                    self.sim_player.try_wallkick(i)
                    
                    new_location = self.piece.location.copy()
                    if new_location.y >= 0:  # Avoid negative indexing
                        self._check_add_to_sets(self.piece.type, new_location, check_placement=True)
                    
                    # Reset piece locations
                    if i != 3:  # Don't need to reset on last rotation
                        self.piece.location = piece_location_copy.copy()
    
    def _optimized_algorithm(self, check_rotations):
        """Faster algorithm with phase-based approach."""
        # Phase 1: Get initial rotations
        phase_2_queue = deque()
        piece_location = self.next_location_queue.popleft()
        piece_location_copy = piece_location.copy()
        
        self.piece.location = piece_location_copy.copy()
        self.piece.coordinates = self.piece.get_self_coords
        phase_2_queue.append(piece_location.copy())
        
        if check_rotations:
            for i in range(1, 4):
                self.sim_player.try_wallkick(i)
                phase_2_queue.append(self.piece.location.copy())
                self._mark_checked(self.piece.type, self.piece.location)
                
                if i != 3:
                    self.piece.location = piece_location_copy.copy()
        
        # Phase 2: Horizontal movement for each rotation
        phase_3_queue = deque()
        while len(phase_2_queue) > 0:
            piece_location = phase_2_queue.popleft()
            phase_3_queue.append(piece_location.copy())
            
            for x_dir in [-1, 1]:
                self.piece.location = piece_location.copy()
                self.piece.coordinates = self.piece.get_self_coords
                self.piece.location.rotation_just_occurred = False
                self.piece.location.rotation_just_occurred_and_used_last_tspin_kick = False
                
                while self.sim_player.can_move(self.piece, x_offset=x_dir):
                    self.piece.location.x += x_dir
                    self.piece.coordinates = self.piece.get_self_coords
                    self._mark_checked(self.piece.type, self.piece.location)
                    phase_3_queue.append(self.piece.location.copy())
        
        # Phase 3: Vertical movement (drop to bottom)
        while len(phase_3_queue) > 0:
            piece_location = phase_3_queue.popleft()
            piece_location_copy = piece_location.copy()
            self.piece.location = piece_location_copy.copy()
            self.piece.coordinates = self.piece.get_self_coords
            
            while self.sim_player.can_move(self.piece, y_offset=1):
                self.piece.location.y += 1
                self.piece.coordinates = self.piece.get_self_coords
                self._mark_checked(self.piece.type, self.piece.location)
            
            # Check these using the normal algorithm and add as placement
            self.next_location_queue.append(self.piece.location.copy())
            self.place_location_queue.append(self.piece.location.copy())
        
        # Phase 4: Use brute force on remaining positions
        self._brute_force_algorithm(check_rotations)
    
    def _harddrop_algorithm(self, check_rotations):
        """Simple algorithm that only considers hard drops."""
        # Phase 1: Get all rotations
        phase_2_queue = deque()
        piece_location = self.next_location_queue.popleft()
        piece_location_copy = piece_location.copy()
        
        self.piece.location = piece_location_copy.copy()
        self.piece.coordinates = self.piece.get_self_coords
        phase_2_queue.append((self.piece.location.x, self.piece.location.y, self.piece.location.rotation))
        
        if check_rotations:
            for i in range(1, 4):
                self.sim_player.try_wallkick(i)
                phase_2_queue.append((self.piece.location.x, self.piece.location.y, self.piece.location.rotation))
                
                if i != 3:
                    self.piece.location = piece_location_copy.copy()
        
        # Phase 2: Horizontal movement
        phase_3_queue = deque()
        while len(phase_2_queue) > 0:
            x, y, rotation = phase_2_queue.popleft()
            phase_3_queue.append((x, y, rotation))
            
            for x_dir in [-1, 1]:
                self.piece.location.x, self.piece.location.y, self.piece.location.rotation = x, y, rotation
                self.piece.coordinates = self.piece.get_self_coords
                
                while self.sim_player.can_move(self.piece, x_offset=x_dir):
                    new_x = self.piece.location.x + x_dir
                    self.piece.location.x = new_x
                    self.piece.coordinates = self.piece.get_self_coords
                    phase_3_queue.append((new_x, y, rotation))
        
        # Phase 3: Hard drop to bottom
        while len(phase_3_queue) > 0:
            x, y, rotation = phase_3_queue.popleft()
            self.piece.location.x, self.piece.location.y, self.piece.location.rotation = x, y, rotation
            self.piece.coordinates = self.piece.get_self_coords
            
            while self.sim_player.can_move(self.piece, y_offset=1):
                self.piece.location.y += 1
                self.piece.coordinates = self.piece.get_self_coords
            
            # Create location object for placement
            final_location = self.piece.location.copy()
            final_location.rotation_just_occurred = False
            final_location.rotation_just_occurred_and_used_last_tspin_kick = False
            
            self.place_location_queue.append(final_location)
    
    def _try_wallkick_via_graphs(self, x, y, rotation, direction, movement_graphs):
        """Wallkick check using precomputed movement graphs instead of per-mino collision.

        Mirrors Player.try_wallkick: tries kicks in table order, first fit wins.
        Returns (x, y, rotation, rotation_just_occurred, used_last_tspin_kick) or None.
        """
        piece_type = self.piece.type
        final_rotation = (rotation + direction) % 4
        kicktable = (i_wallkicks if piece_type == "I" else wallkicks)[rotation][final_rotation]
        graph = movement_graphs[final_rotation]
        row_buffer = self.row_buffer[final_rotation]
        col_buffer = self.col_buffer[final_rotation]

        for kick_index, kick in enumerate(kicktable):
            new_x = x + kick[0]
            new_y = y - kick[1]
            graph_row = new_y + row_buffer
            graph_col = new_x + col_buffer

            if 0 <= graph_row < len(graph) and 0 <= graph_col < len(graph[0]):
                # Graph value 1 or 2 means the position is collision-free
                fits = graph[graph_row][graph_col] != 0
            elif graph_row < 0:
                # Above the graph; positions here can still be valid, check minos directly
                fits = not self.sim_player.collision(
                    [[new_x + col, new_y + row]
                     for col, row in self.mino_coords_dict[piece_type][final_rotation]])
            else:
                # Below or beside the graph is always out of bounds
                fits = False

            if fits:
                if piece_type == "T":
                    used_last = kick_index == len(kicktable) - 1 and direction != 2
                    return (new_x, new_y, final_rotation, True, used_last)
                return (new_x, new_y, final_rotation, False, False)

        return None

    def _convolutional_algorithm(self, check_rotations):
        """
        Advanced convolutional algorithm that finds all moves including spins.
        
        Process:
        1. Convolve grid with each piece rotation to create movement graphs
        2. Start BFS traversal from spawn position within each graph
        3. When hitting graph boundaries, attempt rotations (wallkicks)
        4. If rotation succeeds, jump to corresponding position in new rotation's graph
        5. Continue traversal until all reachable positions found
        """
        axes_of_rotation_dict = {
            "O": 1, "Z": 2, "S": 2, "I": 2,
            "L": 4, "J": 4, "T": 4,
        }
        axes_of_rotation = axes_of_rotation_dict[self.piece.type]
        
        def convolve_grid_with_piece(piece_mask):
            """Convolve grid with piece mask to find valid placement positions.

            Masks are cropped to the minos, so result[row][col] == 1 means the
            piece with its topmost, leftmost mino at (row, col) fits — the graph
            is a (ROWS, COLS) grid in policy coords.
            Out-of-bounds cells are treated as blocked via padding with 1s.
            """
            mask_height, mask_width = piece_mask.shape
            padded = np.pad(self.occupancy, ((0, mask_height - 1), (0, mask_width - 1)),
                            constant_values=1)
            overlaps = signal.correlate2d(padded, piece_mask, mode='valid')
            result = (overlaps == 0).astype(int)
            return result.tolist()
        
        def find_reachable_positions(movement_graph, start_row, start_col, skip_start_placement):
            """BFS to find all positions reachable via left/right/down movement.

            Operates entirely in policy coords (graph indices)."""
            if start_row < 0 or start_row >= len(movement_graph):
                return list(), list()
            if start_col < 0 or start_col >= len(movement_graph[0]):
                return list(), list()
            if movement_graph[start_row][start_col] != 1:
                return list(), list()

            boundary_positions = list()
            placeable_positions = list()
            queue = deque([(start_row, start_col)])

            # Mark as visited at enqueue time so each cell enters the queue once
            movement_graph[start_row][start_col] = 2

            # Don't add the start position to the placement queue
            # because it removes spin information and its placed in the main algorithm

            directions = [(1, 0), (0, 1), (0, -1)]  # down, right, left

            is_first_iteration = True

            while queue:
                row, col = queue.popleft()

                is_boundary = False
                is_placeable = False
                for d_row, d_col in directions:
                    new_row, new_col = row + d_row, col + d_col

                    # Check bounds
                    if (new_row >= len(movement_graph)):
                        is_placeable = True
                        is_boundary = True
                        continue

                    if (new_row < 0 or
                        new_col < 0 or new_col >= len(movement_graph[0])):
                        is_boundary = True
                        continue

                    # If position is reachable and not visited
                    if movement_graph[new_row][new_col] == 1:
                        movement_graph[new_row][new_col] = 2
                        queue.append((new_row, new_col))
                    elif movement_graph[new_row][new_col] == 0:
                        is_boundary = True

                    # Check if it's placeable
                    if d_row == 1 and movement_graph[new_row][new_col] == 0:
                        is_placeable = True

                if is_boundary:
                    boundary_positions.append((row, col))
                if is_placeable: # Don't add first iteration to placeable positions
                    if not (is_first_iteration and skip_start_placement):
                        placeable_positions.append((row, col))

                is_first_iteration = False

            return boundary_positions, placeable_positions

        # Main algorithm starts here
        
        # Step 1: Create convolution graphs for each rotation
        # Store all 4 rotations even if they look the same (different wallkicks/spins)
        movement_graphs = {}
        for rotation in range(4):  # Always store all 4 rotations
            movement_graphs[rotation] = convolve_grid_with_piece(
                PIECE_MASKS[self.piece.type][rotation]
            )
        
        # Step 2: Simple position tracking for rotations
        rotation_queue = deque()

        # Step 3: Start traversal from spawn position
        spawn_x = self.piece.location.x
        spawn_y = self.piece.location.y
        spawn_rotation = self.piece.location.rotation

        # Add spawn position to queue
        rotation_queue.append((spawn_x, spawn_y, spawn_rotation, False, False))

        # Perform rotations initially
        for i in range(1, 4):
            kick_result = self._try_wallkick_via_graphs(spawn_x, spawn_y, spawn_rotation, i, movement_graphs)
            if kick_result is not None and kick_result[1] >= 0:
                rotation_queue.append(kick_result)

        while rotation_queue:
            current_x, current_y, current_rotation, rotation_occurred, used_last_kick = rotation_queue.popleft()

            graph = movement_graphs[current_rotation]
            row_buffer = self.row_buffer[current_rotation]
            col_buffer = self.col_buffer[current_rotation]
            current_row = current_y + row_buffer
            current_col = current_x + col_buffer

            position_is_not_valid = (
                current_row < 0 or current_row >= len(graph) or # Out of vertical bounds
                current_col < 0 or current_col >= len(graph[0]) or # Out of horizontal bounds
                graph[current_row][current_col] == 0 # Not valid position (0 = blocked)
            )

            # Piece checking order of operations:
            if position_is_not_valid: # Out of bounds
                continue

            is_placeable = (
                current_row + 1 >= len(graph) or  # At bottom of grid
                graph[current_row + 1][current_col] == 0  # Blocked below
            )

            already_placed = False

            if is_placeable:
                # Every placeable position is added to the placement queue
                new_location = PieceLocation(current_x, current_y, current_rotation, rotation_occurred, used_last_kick)
                self.place_location_queue.append(new_location)
                already_placed = True

            position_already_processed = graph[current_row][current_col] == 2

            if position_already_processed: # Already processed non-placeable position
                continue

            # Step 4: Find all reachable positions in current rotation's graph
            boundary, placeable = find_reachable_positions(
                graph, current_row, current_col, already_placed
            )

            # Add all reachable positions as valid placements (only if boundary below)
            for row, col in placeable:
                new_location = PieceLocation(col - col_buffer, row - row_buffer, current_rotation, False, False)
                self.place_location_queue.append(new_location)

            # Step 5: Attempt rotations from boundary positions
            if check_rotations:
                for boundary_row, boundary_col in boundary:
                    # Try all 4 rotations from this boundary position (not just axes_of_rotation)
                    boundary_x = boundary_col - col_buffer
                    boundary_y = boundary_row - row_buffer
                    for i in range(1, 4):
                        kick_result = self._try_wallkick_via_graphs(boundary_x, boundary_y, current_rotation, i, movement_graphs)
                        # Avoid negative indexing
                        if kick_result is not None and kick_result[1] >= 0:
                            rotation_queue.append(kick_result)

            # Sort the rotation queue to ensure we process lower rotations first DEBUGGING
            rotation_queue = deque(sorted(rotation_queue, key=lambda x: x[1]))

    
    def _convert_placements_to_policy(self):
        """Convert the placement queue to policy matrix format."""
        policy_matrix = np.zeros(self.POLICY_SHAPE)
        
        for piece_location in self.place_location_queue:
            o = piece_location.rotation

            # Determine T-spin index
            t_spin_index = 0
            if self.piece.type == "T":
                if piece_location.rotation_just_occurred_and_used_last_tspin_kick:
                    t_spin_index = 2
                elif piece_location.rotation_just_occurred:
                    t_spin_index = 1

            # Get policy index; redundant rotations (Z/S/I 2 and 3, O all) share one
            rotation_index = o % len(self.policy_pieces[self.piece.type])
            policy_index = self.policy_piece_to_index[self.piece.type][rotation_index][t_spin_index]

            # Policy coords = topmost, leftmost mino of the placement, so redundant
            # rotations of the same placement encode to the same cell
            policy_row = piece_location.y + self.row_buffer[o]
            policy_col = piece_location.x + self.col_buffer[o]
            policy_matrix[policy_index][policy_row][policy_col] = 1
        
        return policy_matrix


def get_move_matrix(player, algo='brute-force'):
    """
    Main function to generate move matrix for a player.
    
    Args:
        player: The player object containing game state
        algo: Algorithm to use:
            - 'brute-force': Slow but finds every move (100% accuracy)
            - 'faster-but-loss': Faster, 98% accuracy
            - 'harddrop': No spins, just harddrops (fastest)
            - 'convolutional': Advanced algorithm using convolution for complete move finding
            - 'conv-optimized': Optimized convolutional algorithm for better performance
    
    Returns:
        numpy array representing valid moves
        
    Notes:
        - With the true coordinate system, pieces can be placed with negative x values
        - The policy encodes each placement at the (row, col) of its topmost,
          leftmost mino: true coords + coords_to_policy_row/col_buffer[policy_index]
        - Every placement therefore lands within indices [0, ROWS) x [0, COLS)
    """
    # These constants should be imported or passed as parameters in real usage
    # For now, assuming they're globally available or part of player object
    generator = MoveGenerator(
        player=player,
        policy_shape=POLICY_SHAPE,
        policy_pieces=policy_pieces,
        policy_piece_to_index=policy_piece_to_index,
        piece_dict=piece_dict,
        mino_coords_dict=mino_coords_dict,
        rows=ROWS,
        spawn_row=SPAWN_ROW
    )
    
    return generator.generate_moves(algo)