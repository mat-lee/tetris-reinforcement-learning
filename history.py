class History:
    """History class for undo and redo implementation."""
    def __init__(self) -> None:
        self.index = None
        self.boards = []