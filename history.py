import copy

class History:
    """History class for undo and redo implementation."""
    def __init__(self) -> None:
        self.index = None
        self.boards = []
    
    def copy_board_state(self, idx):
        return copy.deepcopy(self.boards[idx])