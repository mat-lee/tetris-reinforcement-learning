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

        # Dimensions: [x, y, rotation] - spin flags are irrelevant for visitation tracking
        checked_shape = (self.POLICY_SHAPE[0], self.POLICY_SHAPE[1], 4)
        self.checked_list = np.zeros(checked_shape, dtype=bool)
    
    def _set_starting_position(self):
        """Set the piece to its starting position."""
        highest_row = self._get_highest_row()
        starting_row = max(highest_row - len(self.piece_dict[self.piece.type]),
                          self.ROWS - self.SPAWN_ROW)
        self.piece.location.y = starting_row

        # Add initial state to queue
        self._add_state_to_queue(self.piece.location.copy())
    
    def _get_highest_row(self):
        """Find the highest occupied row in the grid."""
        grid = self.sim_player.board.grid
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] != 0:
                    return row
        return len(grid)
    
    def _add_state_to_queue(self, piece_location):
        """Add a location to the queue."""
        self.next_location_queue.append(piece_location)

    def _is_placeable(self, piece_location):
        """Check if piece can be placed (can't move down)."""
        self.piece.location = piece_location
        self.piece.coordinates = self.piece.get_self_coords
        return not self.sim_player.can_move(self.piece, y_offset=1)
    
    def _already_checked(self, piece_location):
        """Check if a location has already been processed."""
        # Spin flags are irrelevant for checking if a state has been visited
        x, y, r = piece_location.x, piece_location.y, piece_location.rotation
        return self.checked_list[x + 2, y, r]

    def _mark_checked(self, piece_location):
        """Mark a location as processed."""
        # Spin flags are irrelevant for marking a state as visited
        x, y, r = piece_location.x, piece_location.y, piece_location.rotation
        self.checked_list[x + 2, y, r] = True
    
    def _brute_force_algorithm(self, check_rotations):
        """Exhaustive search algorithm that finds all possible moves."""
        while len(self.next_location_queue) > 0:
            piece_location = self.next_location_queue.popleft()

            self.piece.location = piece_location.copy()
            self.piece.coordinates = self.piece.get_self_coords

            # Check if already visited
            if self._already_checked(piece_location):
                # Still check if placeable for already-visited positions
                if not self.sim_player.can_move(self.piece, y_offset=1):
                    if piece_location not in self.place_location_queue:
                        self.place_location_queue.append(piece_location.copy())
                continue

            self._mark_checked(piece_location)

            # Check if this is a placement location (can't move down)
            if not self.sim_player.can_move(self.piece, y_offset=1):
                self.place_location_queue.append(piece_location.copy())

            # Check left, right, and down moves
            for move in [[1, 0], [-1, 0], [0, 1]]:
                if self.sim_player.can_move(self.piece, x_offset=move[0], y_offset=move[1]):
                    new_location = self.piece.location.copy()
                    new_location.x += move[0]
                    new_location.y += move[1]
                    new_location.rotation_just_occurred = False
                    new_location.rotation_just_occurred_and_used_last_tspin_kick = False
                    self._add_state_to_queue(new_location)

            # Check rotations
            if check_rotations:
                for i in range(1, 4):
                    self.piece.location = piece_location.copy()  # Reset for each kick attempt
                    if self.sim_player.try_wallkick(i):
                        new_location = self.piece.location.copy()
                        if new_location.y >= 0:  # Avoid negative indexing
                            self._add_state_to_queue(new_location)
    
    def _optimized_algorithm(self, check_rotations):
        """A faster, phased algorithm. Marks air cells as visited to constrain brute-force."""
        # Phase 1: Get initial rotations from spawn
        if not self.next_location_queue:
            return

        initial_rotations_q = deque()
        spawn_loc = self.next_location_queue.popleft()
        initial_rotations_q.append(spawn_loc.copy())

        if check_rotations:
            for i in range(1, 4):
                self.piece.location = spawn_loc.copy()
                if self.sim_player.try_wallkick(i):
                    rotated_pos = self.piece.location.copy()
                    initial_rotations_q.append(rotated_pos)
                    self._mark_checked(rotated_pos)

        # Phase 2: Horizontal movement for each rotation
        horizontal_scan_q = deque()
        while len(initial_rotations_q) > 0:
            loc = initial_rotations_q.popleft()
            horizontal_scan_q.append(loc.copy())

            for x_dir in [-1, 1]:
                self.piece.location = loc.copy()
                self.piece.coordinates = self.piece.get_self_coords
                while self.sim_player.can_move(self.piece, x_offset=x_dir):
                    self.piece.location.x += x_dir
                    self.piece.coordinates = self.piece.get_self_coords
                    self._mark_checked(self.piece.location)
                    horizontal_scan_q.append(self.piece.location.copy())

        # Phase 3: Vertical movement (soft drop) - mark the path as visited
        while len(horizontal_scan_q) > 0:
            loc = horizontal_scan_q.popleft()
            self.piece.location = loc.copy()
            self.piece.coordinates = self.piece.get_self_coords

            # Soft drop, marking the path
            while self.sim_player.can_move(self.piece, y_offset=1):
                self._mark_checked(self.piece.location)
                self.piece.location.y += 1
                self.piece.coordinates = self.piece.get_self_coords

            # The final landing spot is a starting point for brute-force
            self._add_state_to_queue(self.piece.location.copy())

        # Phase 4: Use brute force on the collected landing spots
        # The air above has been marked as visited, constraining the search
        self._brute_force_algorithm(check_rotations)
    
    def _harddrop_algorithm(self, check_rotations):
        """Simple algorithm that only considers hard drops from every column/rotation."""
        rotations_to_check = range(4) if check_rotations else [0]
        grid_width = len(self.sim_player.board.grid[0])

        for r in rotations_to_check:
            # Try every column
            for x in range(-2, grid_width):
                # Set piece at top
                spawn_y = self.ROWS - self.SPAWN_ROW
                self.piece.location = PieceLocation(x, spawn_y, r)
                self.piece.coordinates = self.piece.get_self_coords

                # Check if this starting position is valid
                if self.sim_player.collision(self.piece.coordinates):
                    continue

                # Simulate hard drop
                while self.sim_player.can_move(self.piece, y_offset=1):
                    self.piece.location.y += 1
                    self.piece.coordinates = self.piece.get_self_coords

                # The final landing spot is a placement
                final_location = self.piece.location.copy()
                final_location.rotation_just_occurred = False
                final_location.rotation_just_occurred_and_used_last_tspin_kick = False
                self.place_location_queue.append(final_location)
    
    def _convolutional_algorithm(self, check_rotations):
        """
        Finds all piece placements using pre-computed validity maps.

        Key optimization: Collision checks happen once upfront (building validity maps).
        Movement within a rotation state is just array lookups.
        Wallkicks only happen at edges, not at every cell.
        """
        # Step 1: Build validity maps
        validity_maps, x_off, y_off = self._build_validity_maps()

        # Visited tracking per rotation (separate from validity - tracks exploration)
        visited = np.zeros_like(validity_maps, dtype=bool)

        # Step 2: Get spawn position
        spawn_x = self.piece.location.x
        spawn_y = self.piece.location.y
        spawn_r = self.piece.location.rotation

        # Clear the queue since convolution doesn't use it
        self.next_location_queue.clear()

        # Check if spawn is valid
        if not validity_maps[spawn_r, spawn_y + y_off, spawn_x + x_off]:
            return  # Board is topped out

        # Queue stores (x, y, rotation, rotation_just_occurred, used_last_kick)
        exploration_queue = deque()
        exploration_queue.append((spawn_x, spawn_y, spawn_r, False, False))

        # Get wallkick table for this piece
        kick_table = i_wallkicks if self.piece.type == "I" else wallkicks
        rotations_to_try = [1, 2, 3] if check_rotations else []

        while exploration_queue:
            start_x, start_y, rot, rot_just_occurred, used_last_kick = exploration_queue.popleft()

            # If arrived via a kick and this position is immediately stuck, record the
            # kick-flagged placement. Do this BEFORE the visited check: the flood-fill
            # from an earlier rotation state may have already visited this position via
            # sliding, writing a non-kick placement. We still need the kick version.
            if rot_just_occurred:
                map_y_below = start_y + y_off + 1
                stuck = (map_y_below >= validity_maps.shape[1] or
                         not validity_maps[rot, map_y_below, start_x + x_off])
                if stuck:
                    self.place_location_queue.append(PieceLocation(
                        start_x, start_y, rot, True, used_last_kick
                    ))

            # Skip flood-fill if already visited
            if visited[rot, start_y + y_off, start_x + x_off]:
                continue

            # Flood-fill within this rotation state.
            # All placements found here were reached by sliding (not rotating), so
            # rotation_just_occurred = False for all of them.
            edges = self._flood_fill_rotation(
                validity_maps, visited, rot, start_x, start_y, x_off, y_off
            )

            for edge_x, edge_y, is_placeable in edges:
                if is_placeable:
                    self.place_location_queue.append(PieceLocation(
                        edge_x, edge_y, rot, False, False
                    ))

                # Try rotations at edges
                for kick_dir in rotations_to_try:
                    new_rot = (rot + kick_dir) % 4
                    kicks_to_try = kick_table[rot].get(new_rot, [])

                    for kick_idx, (kick_x, kick_y) in enumerate(kicks_to_try):
                        new_x = edge_x + kick_x
                        new_y = edge_y - kick_y  # Kick table Y is inverted

                        map_x = new_x + x_off
                        map_y = new_y + y_off

                        if (0 <= map_x < validity_maps.shape[2] and
                            0 <= map_y < validity_maps.shape[1] and
                            validity_maps[new_rot, map_y, map_x]):

                            new_used_last_kick = (
                                self.piece.type == "T" and
                                kick_dir != 2 and
                                kick_idx == len(kicks_to_try) - 1
                            )

                            # Kicks can push a piece to y<0 when all its minos have
                            # row offsets >= 1. The policy has no row for y<0, so
                            # discard — matches brute-force's y>=0 guard on line 182.
                            if new_y < 0:
                                break

                            if not visited[new_rot, map_y, map_x]:
                                # Unvisited: explore it normally via flood-fill
                                exploration_queue.append((
                                    new_x, new_y, new_rot, True, new_used_last_kick
                                ))
                            else:
                                # Already visited via a non-kick path: the position is
                                # also kick-reachable, so add a kick-flagged placement
                                # if it's immediately stuck (last action = rotation).
                                map_y_below = map_y + 1
                                if (map_y_below >= validity_maps.shape[1] or
                                        not validity_maps[new_rot, map_y_below, map_x]):
                                    self.place_location_queue.append(PieceLocation(
                                        new_x, new_y, new_rot, True, new_used_last_kick
                                    ))
                            break  # First successful kick wins

    def _build_validity_maps(self):
        """
        Pre-compute validity maps for all 4 rotations.
        validity_map[r][y][x] = True means piece can exist at origin (x, y) with rotation r.

        Grid dimensions: (width + 4) x (height + 4) to handle origins outside board bounds.
        X offset: +2 (so origin x=-2 maps to index 0)
        Y offset: +2 (so origin y=-2 maps to index 0)
        """
        grid = self.sim_player.board.grid
        map_width = len(grid[0]) + 4
        map_height = len(grid) + 4
        x_offset = 2
        y_offset = 2

        validity_maps = np.zeros((4, map_height, map_width), dtype=bool)
        piece_type = self.piece.type

        for r in range(4):
            mino_offsets = self.mino_coords_dict[piece_type][r]

            for map_y in range(map_height):
                for map_x in range(map_width):
                    # Convert map coords to actual piece origin coords
                    origin_x = map_x - x_offset
                    origin_y = map_y - y_offset

                    # Compute mino coordinates and check collision
                    coords = [[origin_x + col, origin_y + row] for col, row in mino_offsets]
                    validity_maps[r, map_y, map_x] = not self.sim_player.collision(coords)

        return validity_maps, x_offset, y_offset

    def _flood_fill_rotation(self, validity_maps, visited, rot, start_x, start_y, x_off, y_off):
        """
        Flood-fill within a single rotation state.
        Returns list of edge positions: (x, y, is_placeable)
        """
        edges = []
        stack = [(start_x, start_y)]

        while stack:
            x, y = stack.pop()
            map_x, map_y = x + x_off, y + y_off

            # Bounds check
            if not (0 <= map_x < validity_maps.shape[2] and 0 <= map_y < validity_maps.shape[1]):
                continue

            # Skip if invalid or already visited
            if not validity_maps[rot, map_y, map_x] or visited[rot, map_y, map_x]:
                continue

            # Mark visited
            visited[rot, map_y, map_x] = True

            # Check neighbors
            can_left = (map_x > 0 and validity_maps[rot, map_y, map_x - 1] and
                       not visited[rot, map_y, map_x - 1])
            can_right = (map_x < validity_maps.shape[2] - 1 and
                        validity_maps[rot, map_y, map_x + 1] and
                        not visited[rot, map_y, map_x + 1])
            can_down = (map_y < validity_maps.shape[1] - 1 and
                       validity_maps[rot, map_y + 1, map_x] and
                       not visited[rot, map_y + 1, map_x])

            # Add unvisited valid neighbors to stack
            if can_left:
                stack.append((x - 1, y))
            if can_right:
                stack.append((x + 1, y))
            if can_down:
                stack.append((x, y + 1))

            # Check if this is an edge (blocked in any direction)
            blocked_left = map_x == 0 or not validity_maps[rot, map_y, map_x - 1]
            blocked_right = map_x == validity_maps.shape[2] - 1 or not validity_maps[rot, map_y, map_x + 1]
            blocked_down = map_y == validity_maps.shape[1] - 1 or not validity_maps[rot, map_y + 1, map_x]

            is_edge = blocked_left or blocked_right or blocked_down
            is_placeable = blocked_down  # Can't move down = placement spot

            if is_edge:
                edges.append((x, y, is_placeable))

        return edges

    def _process_kick_placements(self, validity_maps, visited, x_off, y_off, kick_table, rotations_to_try):
        """
        For each existing placement, check if it could also be reached via kick.
        If so, add a T-spin version (with rotation_just_occurred=True).
        """
        new_placements = []

        for placement in self.place_location_queue:
            x, y, rot = placement.x, placement.y, placement.rotation

            # Skip if already has rotation flag
            if placement.rotation_just_occurred:
                continue

            # Check if this position could have been reached via kick
            for kick_dir in rotations_to_try:
                from_rot = (rot - kick_dir) % 4
                kicks = kick_table[from_rot].get(rot, [])

                for kick_idx, (kick_x, kick_y) in enumerate(kicks):
                    from_x = x - kick_x
                    from_y = y + kick_y  # Invert back
                    from_map_x = from_x + x_off
                    from_map_y = from_y + y_off

                    # Check if we visited the source position
                    if (0 <= from_map_x < validity_maps.shape[2] and
                        0 <= from_map_y < validity_maps.shape[1] and
                        visited[from_rot, from_map_y, from_map_x]):

                        # This placement could be reached via kick - add T-spin version
                        used_last_kick = (
                            self.piece.type == "T" and
                            kick_dir != 2 and
                            kick_idx == len(kicks) - 1
                        )

                        new_placements.append(PieceLocation(
                            x, y, rot, True, used_last_kick
                        ))
                        break
                else:
                    continue
                break

        self.place_location_queue.extend(new_placements)
    
    def _convert_placements_to_policy(self):
        """Convert the placement queue to policy matrix format."""
        policy_matrix = np.zeros(self.POLICY_SHAPE)

        # Deduplicate by (x, y, rotation), keeping T-spin version if exists
        unique_placements = {}
        for location in self.place_location_queue:
            key = (location.x, location.y, location.rotation)
            if key not in unique_placements or location.rotation_just_occurred:
                unique_placements[key] = location

        for piece_location in unique_placements.values():
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
            
            # Negative row silently wraps in numpy — this should never happen.
            # (new_col can legitimately be -1 for Z/S/I rotation-3 deduplication,
            # mapping to policy column 1 after the +2 offset.)
            if new_row < 0:
                raise AssertionError(
                    f"Placement row is negative: row={new_row} "
                    f"(piece={self.piece.type}, rotation={o}, x={x}, y={y})"
                )

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