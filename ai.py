from ai_utils import NodeState, PlayerState
from const import *
from player import Player

import random
import treelib

class AI(Player):
    def __init__(self) -> None:
        super().__init__()
        self.draw_coords = (WIDTH/2, 0)

    def make_move(self, human_player, ai_player, player_move=None):
        # Class for picking a move for the AI to make 
        # Initialize the search tree
        keep_tree = False # Option for keeping the usuable part of the old tree
        # You would also have to update all the queues in the tree
        if keep_tree == True:
            if tree == None:
                tree = treelib.Tree()
            else:
                pass
                # Tree = Tree[Players Move]
        else: # Reset the tree
            tree = treelib.Tree()

        # Convert players into player states
        human_state = human_player.to_state()
        ai_state = ai_player.to_state()

        initial_turn = 1 # 0: human move, 1: ai move

        # Create the initial node
        initial_state = NodeState(states=[human_state, ai_state], turn=initial_turn, move=player_move)

        tree.create_node(identifier="root", data=initial_state)

        # To follow along with the tree I'll have two players
        # When a move is made, the player's info becomes the info of the state
        # Then, the new states have the info of the player
        move_players = [Player(), Player()]

        iter = 0
        while iter < MAX_ITER:
            iter += 1

            # Begin at the root node
            node = tree.get_node("root")
            node_state = node.data

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
            
            # Don't update policy, move_list, or generate new nodes if the node is done
            if node_state.is_done == False:

                policy = node_state.get_policy()
                move_list = node_state.get_move_list()

                # Place pieces and generate new leaves
                for move in move_list:
                    # For a new state, the unchanged board can reference the parent board
                    new_state = node.data
                    new_state.states[node_state.turn] = new_state.states[node_state.turn].return_copy()

                    new_state.move = move
                    new_state.make_move(node_state.turn, move, move_players)
                    # new_state.P = policy
                    new_state.P = random.random()

                    new_state.turn = 1 - node_state.turn # 0 -> 1; 1 -> 0

                    tree.create_node(data=new_state, parent=node.identifier)

            value = node_state.get_value()

            # Go back up the tree and updates nodes
            while not node.is_root():
                upwards_id = node.predecessor(tree.identifier)
                node = tree.get_node(upwards_id)

                node_state = node.data
                node_state.N += 1
                node_state.W += value
                node_state.Q = node_state.W / node_state.N

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

        self.force_place_piece(*move)