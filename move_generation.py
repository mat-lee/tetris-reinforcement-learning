from const import *
import numpy as np
from collections import deque
from piece_location import PieceLocation

# Precomputed constants for bitboard operations.
# Map width is always cols+4 = 14; 16 entries covers all valid map_x indices.
_POW2 = tuple(1 << c for c in range(16))
# Packed weights for converting a bool row of width 14 into a bitmask integer.
_BITS_WEIGHTS_14 = np.array([1 << c for c in range(14)], dtype=np.int64)

# Precompute validity-map index arrays once at module load (constants from const.py).
# These depend only on mino_coords_dict and board dimensions, never on game state.
def _precompute_vm_indices():
    map_h = ROWS + 4       # 30
    map_w = COLS + 4       # 14
    y_arange = np.arange(map_h, dtype=np.int32)
    x_arange = np.arange(map_w, dtype=np.int32)
    max_row_off = max(
        row_off
        for pt in mino_coords_dict
        for r in range(4)
        for _co, row_off in mino_coords_dict[pt][r]
    )
    max_col_off = max(
        col_off
        for pt in mino_coords_dict
        for r in range(4)
        for col_off, _ro in mino_coords_dict[pt][r]
    )
    padded_rows = map_h + max_row_off + 2 + 1
    padded_cols = map_w + max_col_off + 2 + 1
    vm_row_idx = {}
    vm_col_idx = {}
    for pt in mino_coords_dict:
        row_arr = np.empty((4, 4, map_h), dtype=np.int32)
        col_arr = np.empty((4, 4, map_w), dtype=np.int32)
        for r in range(4):
            for m, (col_off, row_off) in enumerate(mino_coords_dict[pt][r]):
                row_arr[r, m] = y_arange + row_off + 2
                col_arr[r, m] = x_arange + col_off + 2
        vm_row_idx[pt] = row_arr
        vm_col_idx[pt] = col_arr
    return padded_rows, padded_cols, vm_row_idx, vm_col_idx

_PADDED_ROWS, _PADDED_COLS, _VM_ROW_IDX, _VM_COL_IDX = _precompute_vm_indices()

# Reusable padded board (borders always True/invalid; only interior is overwritten per call).
# Avoids np.ones allocation inside _build_validity_maps on each call.
_PADDED = np.ones((_PADDED_ROWS, _PADDED_COLS), dtype=bool)

# Bitmask for all valid columns in the validity map (COLS+4 = 14 wide).
_R_ALL = (1 << (COLS + 4)) - 1  # 0b11111111111111 = 16383


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
        piece_1 = self.player.piece.type if self.player.piece else None

        # After hold: piece becomes held_piece (if set) or queue[0] (if no held piece)
        if self.player.held_piece is not None:
            piece_2 = self.player.held_piece
        elif self.player.queue.pieces:
            piece_2 = self.player.queue.pieces[0]
        else:
            piece_2 = None

        if piece_1 == piece_2:
            return [piece_1] if piece_1 is not None else []
        return [p for p in [piece_1, piece_2] if p is not None]
    
    def _generate_moves_for_piece(self, piece_type, algorithm):
        """Generate all possible moves for a specific piece type."""
        policy_matrix = np.zeros(self.POLICY_SHAPE)
        
        # Set up simulation state — conv algo never writes to the board, so skip board copy
        if algorithm == 'convolutional':
            self.sim_player = self.player.copy_no_board()
        else:
            self.sim_player = self.player.copy()
        if self.sim_player.piece is None or self.sim_player.piece.type != piece_type:
            self.sim_player.hold_piece()
        
        self.piece = self.sim_player.piece
        if self.piece is None:
            return policy_matrix

        # Determine if piece can rotate
        check_rotations = self.piece.type != "O"

        if algorithm == 'convolutional':
            # Lightweight init: conv algo uses its own visited_bits, no checked_list needed.
            self.next_location_queue = deque()
            self.place_location_queue = []
            self._set_starting_position()
            self._convolutional_algorithm(check_rotations)
            return self._convert_placements_to_policy()

        # Initialize tracking structures (includes checked_list for brute-force)
        self._initialize_tracking_structures()

        # Set starting position
        self._set_starting_position()

        # Generate moves using specified algorithm
        algorithm_map = {
            'brute-force': self._brute_force_algorithm,
            'faster-but-loss': self._optimized_algorithm,
            'harddrop': self._harddrop_algorithm,
        }

        if algorithm not in algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm}")

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
        for i, row in enumerate(grid):
            if any(row):
                return i
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
        Movement within a rotation state is just bitboard integer operations.
        Wallkicks only happen at edges, not at every cell.
        """
        # Step 1: Build validity maps (bitboard representation)
        valid_bits, map_w, map_h, x_off, y_off = self._build_validity_maps()

        # Visited tracking: list of ints per row per rotation (bit c = column c visited)
        visited_bits = [[0] * map_h for _ in range(4)]

        # Step 2: Get spawn position
        spawn_x = self.piece.location.x
        spawn_y = self.piece.location.y
        spawn_r = self.piece.location.rotation
        spawn_map_x = spawn_x + x_off
        spawn_map_y = spawn_y + y_off

        # Clear the queue since convolution doesn't use it
        self.next_location_queue.clear()

        # Check if spawn is valid
        if not (valid_bits[spawn_r][spawn_map_y] & _POW2[spawn_map_x]):
            return  # Board is topped out

        # Queue stores (x, y, rotation, rotation_just_occurred, used_last_kick)
        exploration_queue = deque()
        exploration_queue.append((spawn_x, spawn_y, spawn_r, False, False))

        # Get wallkick table for this piece
        kick_table = i_wallkicks if self.piece.type == "I" else wallkicks
        rotations_to_try = [1, 2, 3] if check_rotations else []
        piece_type = self.piece.type
        is_T = (piece_type == "T")

        # Placements stored as plain tuples (x, y, rot, rot_just_occurred, used_last_kick)
        # to avoid PieceLocation allocation overhead in the hot path.
        # _convert_placements_to_policy unpacks these by index.
        place = self.place_location_queue.append

        # Precompute kick data per rotation (only 4 rotations, not per BFS item).
        kick_data_per_rot = [[] for _ in range(4)]
        for r in range(4):
            kd = []
            for kick_dir in rotations_to_try:
                new_rot = (r + kick_dir) % 4
                kicks = kick_table[r].get(new_rot, [])
                if kicks:
                    kd.append((kick_dir, new_rot, kicks, len(kicks) - 1))
            kick_data_per_rot[r] = kd

        while exploration_queue:
            start_x, start_y, rot, rot_just_occurred, used_last_kick = exploration_queue.popleft()
            map_x = start_x + x_off
            map_y = start_y + y_off

            # If arrived via a kick and this position is immediately stuck, record the
            # kick-flagged placement. Do this BEFORE the visited check: the flood-fill
            # from an earlier rotation state may have already visited this position via
            # sliding, writing a non-kick placement. We still need the kick version.
            map_bit = _POW2[map_x]
            if rot_just_occurred:
                map_y_below = map_y + 1
                stuck = (map_y_below >= map_h or
                         not (valid_bits[rot][map_y_below] & map_bit))
                if stuck:
                    place((start_x, start_y, rot, True, used_last_kick))

            # Skip flood-fill if already visited
            if visited_bits[rot][map_y] & map_bit:
                continue

            kick_data = kick_data_per_rot[rot]
            valid_bits_rot = valid_bits[rot]
            visited_bits_rot = visited_bits[rot]

            # Inline flood-fill within this rotation state (avoids function call +
            # edges list allocation).  Propagates downward row by row.
            if not (valid_bits_rot[map_y] & map_bit):
                continue

            reachable = [0] * map_h
            reachable[map_y] = map_bit

            for fy in range(map_y, map_h):
                r = reachable[fy]
                if not r:
                    continue

                vr = valid_bits_rot[fy]
                unvisited_vr = vr & ~visited_bits_rot[fy]
                while True:
                    new_r = r | ((r << 1) & unvisited_vr) | ((r >> 1) & unvisited_vr)
                    if new_r == r:
                        break
                    r = new_r

                visited_bits_rot[fy] |= r

                next_vr = valid_bits_rot[fy + 1] if fy + 1 < map_h else 0
                blocked_down = r & ~next_vr
                edges_mask = blocked_down | (r & ~(vr << 1)) | (r & ~(vr >> 1))

                # Extract individual edge columns via LSB iteration
                mask = edges_mask
                ey = fy - y_off
                while mask:
                    lsb = mask & (-mask)
                    edge_x = lsb.bit_length() - 1 - x_off
                    is_placeable = bool(blocked_down & lsb)

                    if is_placeable:
                        place((edge_x, ey, rot, False, False))

                    # Try rotations at this edge
                    for kick_dir, new_rot, kicks_to_try, last_kick_idx in kick_data:
                        for kick_idx, (kick_x, kick_y) in enumerate(kicks_to_try):
                            new_x = edge_x + kick_x
                            new_y = ey - kick_y  # Kick table Y is inverted

                            new_map_x = new_x + x_off
                            new_map_y = new_y + y_off

                            new_bit = _POW2[new_map_x] if 0 <= new_map_x < map_w else 0
                            if (new_bit and
                                0 <= new_map_y < map_h and
                                valid_bits[new_rot][new_map_y] & new_bit):

                                new_used_last_kick = (
                                    is_T and
                                    kick_dir != 2 and
                                    kick_idx == last_kick_idx
                                )

                                # Kicks can push a piece to y<0 when all its minos have
                                # row offsets >= 1. The policy has no row for y<0, so
                                # discard — matches brute-force's y>=0 guard on line 182.
                                if new_y < 0:
                                    break

                                if not (visited_bits[new_rot][new_map_y] & new_bit):
                                    # Unvisited: explore it normally via flood-fill
                                    exploration_queue.append((
                                        new_x, new_y, new_rot, True, new_used_last_kick
                                    ))
                                else:
                                    # Already visited via a non-kick path: the position is
                                    # also kick-reachable, so add a kick-flagged placement
                                    # if it's immediately stuck (last action = rotation).
                                    map_y_below = new_map_y + 1
                                    if (map_y_below >= map_h or
                                            not (valid_bits[new_rot][map_y_below] & new_bit)):
                                        place((new_x, new_y, new_rot, True, new_used_last_kick))
                                break  # First successful kick wins

                    mask ^= lsb

                if fy + 1 < map_h:
                    new_reach = r & next_vr & ~visited_bits_rot[fy + 1]
                    if new_reach:
                        reachable[fy + 1] |= new_reach

    def _build_validity_maps(self):
        """
        Pre-compute validity bitboards for all 4 rotations.

        Uses precomputed index arrays (built in __init__) to collapse the
        4-rotation × 4-mino Python loop into a single numpy fancy-index operation.

        The padded board (rows+8, cols+8) = (34, 18) has True = invalid/OOB;
        the real board occupies interior [4:30, 4:14].
        OOB positions fall in the padding (always True = invalid), so no clip needed.

        Returns:
            valid_bits: list of 4 lists-of-ints; valid_bits[r][y] has bit c set if
                        piece with rotation r at origin (c-2, y-2) is valid.
            map_w, map_h: validity map dimensions (14, 30)
            x_offset, y_offset: coordinate offsets (both 2)
        """
        grid = self.player.board.grid
        rows = len(grid)     # 26
        cols = len(grid[0])  # 10
        map_width  = cols + 4  # 14
        map_height = rows + 4  # 30

        # Build padded bool board: True = occupied or OOB (invalid for placement).
        # Reuse module-level array; borders are always True, only interior changes.
        _PADDED[4:4 + rows, 4:4 + cols] = np.asarray(grid, dtype=bool)

        # hit[r, m, y, x] = padded at the board cell covered by mino m
        #                    of rotation r when piece origin is at (x-2, y-2).
        row_idx = _VM_ROW_IDX[self.piece.type]  # (4, 4, 30)
        col_idx = _VM_COL_IDX[self.piece.type]  # (4, 4, 14)
        hit = _PADDED[row_idx[:, :, :, None], col_idx[:, :, None, :]]  # (4, 4, 30, 14)

        # Position invalid if ANY mino is occupied or OOB; pack validity into bitmasks.
        # Vectorize across all 4 rotations at once (avoids Python for-loop).
        invalid = hit.any(axis=1)          # (4, 30, 14) bool
        valid_bits = (~invalid @ _BITS_WEIGHTS_14).tolist()  # (4, 30) int64 → list of lists

        return valid_bits, map_width, map_height, 2, 2

    def _flood_fill_rotation(self, valid_bits_rot, visited_bits_rot,
                             map_h, map_w, start_x, start_y, x_off, y_off):
        """
        Bitboard scanline flood fill within a single rotation state.

        valid_bits_rot:  list of map_h ints; bit c set ↔ column c is valid at that row.
        visited_bits_rot: list of map_h ints (mutated in-place); tracks visited columns.

        Movement model: left, right, and down only (no upward movement).
        Scanline top-to-bottom; horizontal expansion via bit ops per row.

        Returns list of edge positions: (x, y, is_placeable)
        """
        edges = []

        start_map_x = start_x + x_off
        start_map_y = start_y + y_off

        # Bounds / validity / visited check at entry point
        if not (0 <= start_map_x < map_w and 0 <= start_map_y < map_h):
            return edges
        start_bit = _POW2[start_map_x]
        if not (valid_bits_rot[start_map_y] & start_bit):
            return edges
        if visited_bits_rot[start_map_y] & start_bit:
            return edges

        # Per-row reachable bitmasks (propagated downward from start_map_y)
        reachable = [0] * map_h
        reachable[start_map_y] = start_bit

        for map_y in range(start_map_y, map_h):
            r = reachable[map_y]
            if not r:
                continue

            vr = valid_bits_rot[map_y]
            # Expand only into unvisited valid cells (horizontally).
            # & _R_ALL is redundant since vr has at most bits 0..(map_w-1).
            unvisited_vr = vr & ~visited_bits_rot[map_y]
            while True:
                new_r = r | ((r << 1) & unvisited_vr) | ((r >> 1) & unvisited_vr)
                if new_r == r:
                    break
                r = new_r

            # Mark this row's newly reached cells as visited
            visited_bits_rot[map_y] |= r

            # Edge detection — based on validity only (not visited status).
            # r is already bounded by _R_ALL so the final & _R_ALL is redundant.
            next_vr = valid_bits_rot[map_y + 1] if map_y + 1 < map_h else 0
            blocked_down  = r & ~next_vr
            edges_mask = blocked_down | (r & ~(vr << 1)) | (r & ~(vr >> 1))

            # Extract individual edge columns via LSB iteration
            mask = edges_mask
            y = map_y - y_off
            while mask:
                lsb = mask & (-mask)
                c = lsb.bit_length() - 1
                is_placeable = bool(blocked_down & lsb)
                edges.append((c - x_off, y, is_placeable))
                mask ^= lsb

            # Propagate reachable cells downward (skip already-visited cells below)
            if map_y + 1 < map_h:
                new_reach = r & next_vr & ~visited_bits_rot[map_y + 1]
                if new_reach:
                    reachable[map_y + 1] |= new_reach

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
        """Convert the placement queue to policy matrix format.

        Accepts either PieceLocation objects (brute-force path) or plain 5-tuples
        (x, y, rot, rotation_just_occurred, used_last_kick) (convolutional path).
        """
        policy_matrix = np.zeros(self.POLICY_SHAPE)

        piece_type = self.piece.type
        is_T = (piece_type == "T")
        is_ZSI = piece_type in ("Z", "S", "I")
        piece_rotations = self.policy_pieces[piece_type]
        n_rotations = len(piece_rotations)
        piece_policy_idx = self.policy_piece_to_index[piece_type]

        # Detect whether placements are tuples (conv path) or PieceLocations (brute-force).
        # Checking the first element is enough — all placements come from one algorithm.
        use_tuple = self.place_location_queue and isinstance(
            self.place_location_queue[0], tuple
        )

        # Deduplicate by (x, y, rotation), keeping T-spin version if exists.
        unique_placements = {}
        if use_tuple:
            for loc in self.place_location_queue:
                key = loc[:3]
                if key not in unique_placements or loc[3]:
                    unique_placements[key] = loc
        else:
            for loc in self.place_location_queue:
                key = (loc.x, loc.y, loc.rotation)
                if key not in unique_placements or loc.rotation_just_occurred:
                    unique_placements[key] = loc

        # Collect (policy_index, row, col) for batch numpy assignment at the end.
        pi_list = []
        ri_list = []
        ci_list = []

        if use_tuple:
            for x, y, o, rot_occurred, used_last_kick in unique_placements.values():
                t_spin_index = 0
                if is_T and rot_occurred:
                    t_spin_index = 2 if used_last_kick else 1

                rotation_index = o % n_rotations
                policy_index = piece_policy_idx[rotation_index][t_spin_index]

                new_col = x
                new_row = y
                if is_ZSI:
                    if o == 2:
                        new_row += 1
                    elif o == 3:
                        new_col -= 1

                if new_row < 0:
                    raise AssertionError(
                        f"Placement row is negative: row={new_row} "
                        f"(piece={piece_type}, rotation={o}, x={x}, y={y})"
                    )

                pi_list.append(policy_index)
                ri_list.append(new_row)
                ci_list.append(new_col + 2)
        else:
            for loc in unique_placements.values():
                x, y, o = loc.x, loc.y, loc.rotation
                rot_occurred = loc.rotation_just_occurred
                used_last_kick = loc.rotation_just_occurred_and_used_last_tspin_kick

                t_spin_index = 0
                if is_T and rot_occurred:
                    t_spin_index = 2 if used_last_kick else 1

                rotation_index = o % n_rotations
                policy_index = piece_policy_idx[rotation_index][t_spin_index]

                new_col = x
                new_row = y
                if is_ZSI:
                    if o == 2:
                        new_row += 1
                    elif o == 3:
                        new_col -= 1

                if new_row < 0:
                    raise AssertionError(
                        f"Placement row is negative: row={new_row} "
                        f"(piece={piece_type}, rotation={o}, x={x}, y={y})"
                    )

                pi_list.append(policy_index)
                ri_list.append(new_row)
                ci_list.append(new_col + 2)

        if pi_list:
            policy_matrix[pi_list, ri_list, ci_list] = 1

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