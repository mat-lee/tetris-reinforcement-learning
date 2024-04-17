import random

minos = "ZLOSIJT"

class Queue:
    def __init__(self, pieces=None):
        self.pieces = pieces if pieces is not None else []
    
    def add_bag(self, bag):
        self.pieces += bag
    
    def copy(self):
        new_queue = Queue()
        new_queue.pieces = self.pieces[:]
        
        return new_queue

    @staticmethod
    def generate_bag():
        mino_list = list(minos)
        random.shuffle(mino_list)
        return mino_list