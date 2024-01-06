class Queue:
    def __init__(self, pieces=None):
        self.pieces = pieces if pieces is not None else []
    
    def add_bag(self, bag):
        self.pieces += bag