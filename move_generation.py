from const import *
import numpy as np
from collections import deque
from piece_location import PieceLocation


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
        
    def generate_moves(self, algorithm='brute-force'):
        """Main entry point for move generation."""
        new_policy = np.zeros(self.POLICY_SHAPE)
        
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
        
        # Dimensions: [x, y, rotation, t_spin_state]
        t_spin_states = 3 if self.piece.type == "T" else 1
        checked_shape = (self.POLICY_SHAPE[0], self.POLICY_SHAPE[1], 4, t_spin_states)
        self.checked_list = np.zeros(checked_shape, dtype=int)
    
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
    
    def _already_checked(self, piece_type, piece_location):
        """Check if a location has already been processed."""
        if piece_type == "T":
            index = 0
            if piece_location.rotation_just_occurred:
                index = 1
            if piece_location.rotation_just_occurred_and_used_last_tspin_kick:
                index = 2
            return self.checked_list[piece_location.x + 2][piece_location.y][piece_location.rotation][index] == 1
        return self.checked_list[piece_location.x + 2][piece_location.y][piece_location.rotation] == 1
    
    def _mark_checked(self, piece_type, piece_location):
        """Mark a location as processed."""
        if piece_type == "T":
            index = 0
            if piece_location.rotation_just_occurred:
                index = 1
            if piece_location.rotation_just_occurred_and_used_last_tspin_kick:
                index = 2
            self.checked_list[piece_location.x + 2][piece_location.y][piece_location.rotation][index] = 1
        else:
            self.checked_list[piece_location.x + 2][piece_location.y][piece_location.rotation] = 1
    
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
        
        def create_piece_mask(piece_type, rotation):
            """Create a binary mask for the piece at given rotation, preserving original coordinates"""
            # Use the piece dictionary which already has the proper matrix representation
            # This preserves the original piece coordinates and matrix structure
            mask = self.piece_dict[piece_type]
            
            # Rotate the mask to match the requested rotation
            rotated_mask = [row[:] for row in mask]  # Deep copy
            for _ in range(rotation):
                rotated_mask = np.rot90(rotated_mask, 3).tolist()  # Rotate 90 degrees clockwise
                
            return rotated_mask
        
        def convolve_grid_with_piece(grid, piece_mask):
            """Convolve grid with piece mask to find valid placement positions."""
            grid_height = len(grid)
            grid_width = len(grid[0])
            mask_height = len(piece_mask)
            mask_width = len(piece_mask[0])
            
            # Result includes buffer for negative x positions (-2 to grid_width-1)
            result = np.zeros((POLICY_SHAPE[1], POLICY_SHAPE[2]), dtype=int)  # +2 buffer for x=-2 to x=grid_width-1
            
            # Scan all possible positions
            for grid_row in range(POLICY_SHAPE[1]):
                for grid_col in range(-2, -2 + POLICY_SHAPE[2]):  # Allow negative x
                    
                    # Check if piece can be placed at this position
                    can_place = True
                    for mask_row in range(mask_height):
                        for mask_col in range(mask_width):
                            if piece_mask[mask_row][mask_col] != 0:  # If piece occupies this cell
                                actual_row = grid_row + mask_row
                                actual_col = grid_col + mask_col
                                
                                # Check bounds and collisions
                                if (actual_col < 0 or actual_col >= grid_width or
                                    actual_row < 0 or actual_row >= grid_height or
                                    grid[actual_row][actual_col] != 0):
                                    can_place = False
                                    break
                        if not can_place:
                            break
                    
                    if can_place:
                        result[grid_row][grid_col + 2] = 1  # +2 buffer offset
                        
            return result.tolist()
        
        def find_reachable_positions(movement_graph, start_x, start_y, skip_start_placement):
            """BFS to find all positions reachable via left/right/down movement."""
            if start_y < 0 or start_y >= len(movement_graph):
                return list(), list()
            if start_x + 2 < 0 or start_x + 2 >= len(movement_graph[0]):
                return list(), list()
            if movement_graph[start_y][start_x + 2] != 1:
                return list(), list()
                
            boundary_positions = list()
            placeable_positions = list()
            queue = deque([(start_x, start_y)])

            # Don't add the start position to the placement queue 
            # because it removes spin information and its placed in the main algorithm
            
            directions = [(0, 1), (1, 0), (-1, 0)]  # down, right, left

            is_first_iteration = True
            
            while queue:
                x, y = queue.popleft()
                
                # Skip if already processed or invalid
                if (x + 2 < 0 or x + 2 >= len(movement_graph[0]) or 
                    y < 0 or y >= len(movement_graph) or
                    movement_graph[y][x + 2] != 1):
                    continue
                
                # Mark as visited in the graph
                movement_graph[y][x + 2] = 2
                
                is_boundary = False
                is_placeable = False
                for dx, dy in directions:
                    new_x, new_y = x + dx, y + dy
                    
                    # Check bounds
                    if (new_y >= len(movement_graph)):
                        is_placeable = True
                        is_boundary = True
                        continue

                    if (new_y < 0 or
                        new_x + 2 < 0 or new_x + 2 >= len(movement_graph[0])):
                        is_boundary = True
                        continue
                        
                    # If position is reachable and not visited
                    if movement_graph[new_y][new_x + 2] == 1:
                        if (new_x, new_y) not in queue:
                            queue.append((new_x, new_y))
                    elif movement_graph[new_y][new_x + 2] == 0:
                        is_boundary = True
                    
                    # Check if it's placeable
                    if dy == 1 and movement_graph[new_y][new_x + 2] == 0:
                        is_placeable = True 
                
                if is_boundary:
                    boundary_positions.append((x, y))
                if is_placeable: # Don't add first iteration to placeable positions
                    if not (is_first_iteration and skip_start_placement):
                        placeable_positions.append((x, y))
                
                is_first_iteration = False
                    
            return boundary_positions, placeable_positions

        # Main algorithm starts here
        
        # Step 1: Create convolution graphs for each rotation
        # Store all 4 rotations even if they look the same (different wallkicks/spins)
        movement_graphs = {}
        for rotation in range(4):  # Always store all 4 rotations
            piece_mask = create_piece_mask(self.piece.type, rotation)
            movement_graphs[rotation] = convolve_grid_with_piece(
                self.sim_player.board.grid, piece_mask
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
            self.piece.location.x = spawn_x
            self.piece.location.y = spawn_y
            self.piece.location.rotation = spawn_rotation
            self.piece.coordinates = self.piece.get_self_coords

            if self.sim_player.try_wallkick(i):
                if self.piece.location.y >= 0:
                    rotation_queue.append((self.piece.location.x, self.piece.location.y, self.piece.location.rotation, self.piece.location.rotation_just_occurred, 
                                            self.piece.location.rotation_just_occurred_and_used_last_tspin_kick))

        while rotation_queue:
            current_x, current_y, current_rotation, rotation_occurred, used_last_kick = rotation_queue.popleft()

            position_is_not_valid = (
                current_y < 0 or current_y >= len(movement_graphs[current_rotation]) or # Out of vertical bounds
                current_x + 2 < 0 or current_x + 2 >= len(movement_graphs[current_rotation][0]) or # Out of horizontal bounds
                movement_graphs[current_rotation][current_y][current_x + 2] == 0 # Not valid position (0 = blocked)
            )

            # Piece checking order of operations:
            if position_is_not_valid: # Out of bounds
                continue

            is_placeable = (
                current_y + 1 >= len(movement_graphs[current_rotation]) or  # At bottom of grid
                current_x + 2 < 0 or current_x + 2 >= len(movement_graphs[current_rotation][0]) or  # Out of bounds
                movement_graphs[current_rotation][current_y + 1][current_x + 2] == 0  # Blocked below
            )

            already_placed = False

            if is_placeable:
                # Every placeable position is added to the placement queue
                new_location = PieceLocation(current_x, current_y, current_rotation, rotation_occurred, used_last_kick)
                self.place_location_queue.append(new_location)
                already_placed = True
                
            position_already_processed = movement_graphs[current_rotation][current_y][current_x + 2] == 2

            if position_already_processed: # Already processed non-placeable position
                continue

            # Step 4: Find all reachable positions in current rotation's graph
            boundary, placeable = find_reachable_positions(
                movement_graphs[current_rotation], current_x, current_y, already_placed
            )
            
            # Add all reachable positions as valid placements (only if boundary below)
            for x, y in placeable:
                new_location = PieceLocation(x, y, current_rotation, False, False)
                self.place_location_queue.append(new_location)
            
            # Step 5: Attempt rotations from boundary positions
            if check_rotations:
                for boundary_x, boundary_y in boundary:
                    # Try all 4 rotations from this boundary position (not just axes_of_rotation)
                    for i in range(1, 4):
                        # Try all 4 rotations for wallkick purposes
                        self.piece.location.x = boundary_x
                        self.piece.location.y = boundary_y
                        self.piece.location.rotation = current_rotation
                        self.piece.coordinates = self.piece.get_self_coords

                        if self.sim_player.try_wallkick(i):
                            # Avoid negative indexing
                            if self.piece.location.y >= 0:
                                rotation_queue.append((self.piece.location.x, self.piece.location.y, self.piece.location.rotation, self.piece.location.rotation_just_occurred, 
                                                        self.piece.location.rotation_just_occurred_and_used_last_tspin_kick))

            # Sort the rotation queue to ensure we process lower rotations first DEBUGGING
            rotation_queue = deque(sorted(rotation_queue, key=lambda x: x[1]))
    
    def _convert_placements_to_policy(self):
        """Convert the placement queue to policy matrix format."""
        policy_matrix = np.zeros(self.POLICY_SHAPE)
        
        for piece_location in self.place_location_queue:
            x = piece_location.x
            y = piece_location.y
            o = piece_location.rotation
            
            # Determine T-spin index
            t_spin_index = 0
            if self.piece.type == "T":
                if piece_location.rotation_just_occurred_and_used_last_tspin_kick:
                    t_spin_index = 2
                elif piece_location.rotation_just_occurred:
                    t_spin_index = 1
            
            # Get policy index
            rotation_index = o % len(self.policy_pieces[self.piece.type])
            policy_index = self.policy_piece_to_index[self.piece.type][rotation_index][t_spin_index]
            
            # Adjust coordinates for certain pieces
            new_col = x
            new_row = y
            if self.piece.type in ["Z", "S", "I"]:
                # For those pieces, rotation 2 is the same as rotation 0 but moved one down
                if o == 2:
                    new_row += 1
                # For those pieces, rotation 3 is the same as rotation 1 but moved one to the left
                if o == 3:
                    new_col -= 1
            
            # Set policy value (account for x-coordinate buffer)
            policy_matrix[policy_index][new_row][new_col + 2] = 1
        
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
        - With this coordinate system, pieces can be placed with negative x values
        - To avoid negative indexing, the list of moves is shifted by 2
        - It encodes moves from -2 to 8 as indices 0 to 10
        - Possible locations account for blocks with negative x:
            * O piece: 0 to 8
            * Any 3x3 piece: -1 to 8  
            * I piece: -2 to 8
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