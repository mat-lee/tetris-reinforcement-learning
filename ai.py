from const import *
from player import Player

import random
import treelib

def MCTS(active_player, other_player):
    # Class for picking a move for the AI to make 
    # Initialize the search tree
    tree = treelib.Tree()

    initial_turn = 0 # 0: own move, 1: other move

    game_copy = [active_player.copy(), other_player.copy()]
    # Restrict previews
    for player in game_copy:
        while len(player.queue.pieces) > PREVIEWS:
            player.queue.pieces.pop(-1)

    # Create the initial node
    initial_state = NodeState(players=game_copy, 
                                turn=initial_turn, move=None)

    tree.create_node(identifier="root", data=initial_state)

    MAX_DEPTH = 0
    iter = 0
    while iter < MAX_ITER:
        iter += 1

        # Begin at the root node
        node = tree.get_node("root")
        node_state = node.data

        DEPTH = 0

        # Go down the tree using Q+U until you get to a leaf node
        while not node.is_leaf():
            child_ids = node.successors(tree.identifier)
            max_child_score = 0
            max_child_id = None
            sum_n = 0
            for child_id in child_ids:
                sum_n += tree.get_node(child_id).data.N
            for child_id in child_ids:
                child_data = tree.get_node(child_id).data
                child_score = child_data.P*sum_n/(1+child_data.N)
                if child_score > max_child_score:
                    max_child_score = child_score
                    max_child_id = child_id
            
            node = tree.get_node(max_child_id)
            node_state = node.data

            DEPTH += 1
            if DEPTH > MAX_DEPTH:
                MAX_DEPTH = DEPTH
                    
        # Don't update policy, move_list, or generate new nodes if the node is done
        if node_state.is_done == False:

            value, policy = evaluate(node_state)
            move_list = get_move_list(node_state.players[node_state.turn])

            # Place pieces and generate new leaves
            for move in move_list:
                copied_players = []
                for player in node_state.players:
                    copied_players.append(player.copy())
                new_state = NodeState(players=copied_players, 
                                        turn=node_state.turn,
                                        move=move)

                new_state.make_move(node_state.turn, move)
                # new_state.P = policy
                new_state.P = random.random()

                new_state.turn = 1 - node_state.turn # 0 -> 1; 1 -> 0

                tree.create_node(data=new_state, parent=node.identifier)

        # Go back up the tree and updates nodes
        while not node.is_root():
            upwards_id = node.predecessor(tree.identifier)
            node = tree.get_node(upwards_id)

            node_state = node.data
            node_state.N += 1
            node_state.W += value
            node_state.Q = node_state.W / node_state.N

    print(MAX_DEPTH)

    # Choose a move
    root = tree.get_node("root")
    root_children_id = root.successors(tree.identifier)
    max_n = 0
    max_id = None

    for root_child_id in root_children_id:
        root_child = tree.get_node(root_child_id)
        root_child_n = root_child.data.N
        if root_child_n > max_n:
            max_n = root_child_n
            max_id = root_child.identifier

    move = tree.get_node(max_id).data.move

    return move

def evaluate(game):
    return random.random(), 1

class NodeState():
    """Node class for storing the game in the tree.
    
    Similar to the game class, but is more specialized for AI functions and has different variables"""

    def __init__(self, players=None, turn=None, move=None) -> None:
        self.players = players
        self.turn = turn
        self.move = move

        # Search tree variables
        # Would be stored in each edge, but this is essentially the same thing
        self.N = 1
        self.W = 0
        self.Q = 0
        self.P = 0

        self.value = None
        self.policy = None
        self.move_list = None

        self.is_done = False

#class GameState():
    """Class for storing game information inside nodes."""
    
    def check_if_done(self):
        for player in self.players:
            if player.game_over == True:
                self.is_done = True
    
    def make_move(self, turn, move): # e^-5
        player = self.players[turn]
        other_player = self.players[turn - 1]

        player.force_place_piece(*move)

        # Checks for sending garbage, sends garbage, and canceling
        while len(player.garbage_to_send) > 0 and len(player.garbage_to_receive) > 0: # Cancel garbage
            # Remove first elements
            player.garbage_to_send.pop(0)
            player.garbage_to_receive.pop(0)

        if len(player.garbage_to_send) > 0:
            other_player.garbage_to_receive += player.garbage_to_send # Send garbage
            player.garbage_to_send = [] # Stop sending garbage
        
        elif len(player.garbage_to_receive) > 0:
            player.spawn_garbage()
        
        player.create_next_piece()

        # Check if the game is over:
        self.check_if_done()

        # Stop searching if the other player doesn't have a piece
        if len(other_player.queue.pieces) == 0 and other_player.piece == None:
            self.is_done = True

def get_move_list(player): # e^-3
    # Using a list of all possible locations the piece can reach
    # Using a queue to determine which boards to check

    # Possible locations needs to be larger to acount for blocks with negative x and y
    # O piece        0 to 8
    # Any 3x3 piece -1 to 8
    # I piece       -2 to 8

    def get_highest_row(grid):
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] != 0:
                    highest_row = row
                    return highest_row
        
        # Else return the floor
        return len(grid)

    # Load the state
    sim_player = Player()
    sim_player.board = player.board.copy()
    sim_player.piece = player.piece.copy()

    piece = sim_player.piece

    # On the left hand side, blocks can have negative x
    buffer = (2 if piece.type == "I" else (0 if piece.type == "O" else 1))

    # No piece can be placed in the bottom row; ROWS - 1
    possible_piece_locations = [[[False for o in range (4)] for x in range(COLS + buffer - 1)] for y in range(ROWS - 1)]
    next_location_queue = []
    locations = []

    # Start the piece at the highest point it can be placed
    highest_row = get_highest_row(sim_player.board.grid)
    starting_row = max(highest_row - len(piece.matrix), 0)
    piece.y_0 = starting_row

    x = piece.x_0
    y = piece.y_0
    o = piece.rotation

    piece.update_rotation()

    next_location_queue.append((x, y, o))

    # Search through the queue
    while len(next_location_queue) > 0:
        piece.x_0, piece.y_0, piece.rotation = next_location_queue[0]

        piece.update_rotation()

        possible_piece_locations[piece.y_0][piece.x_0 + buffer][piece.rotation] = True

        for move in [[1, 0], [-1, 0], [0, 1]]:
            x = piece.x_0 + move[0]
            y = piece.y_0 + move[1]
            o = piece.rotation

            if sim_player.can_move(x_offset=move[0], y_offset=move[1]): # Check this first to avoid index errors
                if (possible_piece_locations[y][x + buffer][o] == False 
                    and (x, y, o) not in next_location_queue
                    and y >= 0): # Avoid negative indexing

                    next_location_queue.append((x, y, o))


        for i in range(1, 4):
            sim_player.try_wallkick(i)

            x = piece.x_0
            y = piece.y_0
            o = piece.rotation

            if (possible_piece_locations[y][x + buffer][o] == False
                and (x, y, o) not in next_location_queue
                and y >= 0): # Avoid negative indexing
                next_location_queue.append((x, y, o))

            piece.x_0, piece.y_0, piece.rotation = next_location_queue[0]

            piece.update_rotation()

        next_location_queue.pop(0)

    # Remove entries that can move downwards
    for o in range(4): # Smallest number of operations
        sim_player.piece.rotation = o
        sim_player.piece.update_rotation()

        for x in range(COLS + buffer - 1):
            sim_player.piece.x_0 = x - buffer

            for y in reversed(range(ROWS - 1)):
                if possible_piece_locations[y][x][o] == True:
                    sim_player.piece.y_0 = y
                    if not sim_player.can_move(y_offset=1):
                        locations.append((x - buffer, y, o))

    return locations 

# Using deepcopy:                  100 iter in 36.911 s
# Using copy functions in classes: 100 iter in 1.658 s
# Many small changes:              100 iter in 1.233 s