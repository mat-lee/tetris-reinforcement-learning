"""Stores statistics for the game."""
import math

class Stats:
    def __init__(self):
        self.pieces = 0
        self.lines_cleared = 0
        self.b2b = -1
        self.b2b_level = 0
        self.combo = 0
        self.lines_sent = 0

    def update_b2b(self):
        b2b_chart = [[-1, 0],
                     [1, 1],
                     [3, 2], 
                     [8, 3],
                     [24, 4],
                     [67, 5],
                     [185, 6],
                     [504, 7],
                     [1370, 8]]
        
        for list in b2b_chart:
            if (self.b2b >= list[0]) and (self.b2b_level < list[1]):
                self.b2b_level = list[1]

    def update_stats(self, rows_cleared, is_tspin, is_mini, is_all_clear):
        attack = 0

        # doens't look very pretty now does it
        if rows_cleared == 0:
            self.combo = 0
        else:
            if rows_cleared == 1:
                if is_tspin == False:
                    self.b2b = -1
                    attack = math.floor(0.5 + 0.25 * self.combo)
                else:
                    if is_mini == True:
                        if self.b2b == 0:
                            attack = math.floor(0.5 + 0.25 * self.combo)
                        else:
                            self.update_b2b()
                            attack = math.floor(self.b2b_level * (1 + 0.25 * self.combo))
                    else:
                        self.b2b += 1
                        self.update_b2b()
                        attack = math.floor((2 + self.b2b_level) * (1 + 0.25 * self.combo))
            elif rows_cleared == 2:
                if is_tspin == False:
                    self.b2b = 0
                    attack = math.floor(1 + 0.25 * self.combo)
                else:
                    if is_mini == True:
                        self.update_b2b()
                        attack = math.floor((1 + self.b2b_level) * (1 + 0.25 * self.combo))
                    else:
                        self.b2b += 1
                        self.update_b2b()
                        attack = math.floor((1 + 0.25 * self.b2b_level) * (4 + 1 * self.combo))
            elif rows_cleared == 3:
                if is_tspin == False:
                    self.b2b = 0
                    attack = math.floor(2 + 0.5 * self.combo)
                else:
                    self.b2b += 1
                    self.update_b2b()
                    attack = math.floor((1 + 1/6 * self.b2b_level) * (6 + 1.5 * self.combo))
            elif rows_cleared == 4:
                self.b2b += 1
                self.update_b2b()
                attack += math.floor((1 + 0.25 * self.b2b_level) * (4 + 1 * self.combo))
            self.combo += 1
        
        if is_all_clear:
            attack += 10

        self.lines_sent += attack

        self.lines_cleared += rows_cleared

    # for making sure I implemented garbage sending correctly...
    # don't look please
    # it does work though

    '''def make_chart(self):
        combo_max = 5
        types = [
            [1, False, False, 0],
            [2, False, False, 0], 
            [3, False, False, 0], 
            [4, False, False, 0], 
            [1, True, True, 0], 
            [1, True, False, 0], 
            [2, True, True, 0], 
            [2, True, False, 0], 
            [3, True, False, 0],
            [4, False, False, 1], 
            [1, True, True, 1], 
            [1, True, False, 1], 
            [2, True, True, 1], 
            [2, True, False, 1], 
            [3, True, False, 1],
            [4, False, False, 2], 
            [1, True, True, 2], 
            [1, True, False, 2], 
            [2, True, True, 2], 
            [2, True, False, 2], 
            [3, True, False, 2],
            ]
        chart = [[0] * combo_max for x in range(len(types))]
        for i in range(5):
            for j, type in enumerate(types):
                chart[j][i] = Stats()
                chart[j][i].combo = i
                chart[j][i].b2b = type[3]
                chart[j][i].update_stats(type[0],type[1],type[2])
                chart[j][i] = chart[j][i].lines_sent
        print(chart)'''