"""Stores statistics for the game."""
import math

class Stats:
    def __init__(self, ruleset):
        self.pieces = 0
        self.lines_cleared = 0
        self.b2b = -1
        self.b2b_level = 0
        self.combo = 0
        self.lines_sent = 0
        self.attack_text = ''

        self.ruleset = ruleset

    def copy(self):
        new_stats = Stats(self.ruleset)
        new_stats.pieces = self.pieces
        new_stats.lines_cleared = self.lines_cleared
        new_stats.b2b = self.b2b
        new_stats.b2b_level = self.b2b_level
        new_stats.combo = self.combo
        new_stats.lines_sent = self.lines_sent
        new_stats.attack_text = self.attack_text
        
        return new_stats

    def update_b2b_level(self):
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

    def get_attack(self, rows_cleared, is_tspin, is_mini, is_all_clear):
        # Returns attack + functions as update stats for now
        attack = 0

        if self.ruleset == 's1':
            # I can get a general formula for attacks larger than 1 row
            if rows_cleared == 0:
                self.combo = 0
            else:
                is_b2b = False
                if rows_cleared == 4 or is_tspin == True:
                    is_b2b = True
                
                if is_b2b == True:
                    self.b2b += 1
                else:
                    self.b2b = -1
                
                self.update_b2b_level()

                if rows_cleared == 1:
                    if is_tspin == False:
                        self.b2b = -1
                        attack += math.floor(0.5 + 0.25 * self.combo) ### formula exception
                    else:
                        if is_mini == True:
                            if self.b2b <= 0 and self.b2b_level <= 0:
                                attack += math.floor(0.5 + 0.25 * self.combo) ### formula exception
                            else:
                                attack += math.floor(self.b2b_level * (1 + 0.25 * self.combo)) ### formula exception
                        else:
                            attack += math.floor((2 + self.b2b_level) * (1 + 0.25 * self.combo))
                
                else: attack += math.floor((1 + 0.25 * self.combo) * 
                                            (2 * rows_cleared * is_tspin * (-3/4 * is_mini + 1)
                                                + 2**(rows_cleared-2) * (1 - is_tspin)
                                                + self.b2b_level * (is_tspin or rows_cleared == 4))) # General formula for 2-4 rows cleared

                self.combo += 1

                if is_all_clear:
                    attack += 10
        
        elif self.ruleset == 's2':
            # I can get a general formula for attacks larger than 1 row
            if rows_cleared == 0:
                self.combo = 0
            else:
                is_b2b = False
                if is_tspin == True or is_mini == True or rows_cleared == 4:
                    is_b2b = True
                
                if is_b2b == True:
                    self.b2b += 1

                elif is_all_clear: # Don't lower b2b if all clear
                    attack += 5
                    self.b2b += 1

                else:
                    # Surge
                    if self.b2b >= 4:
                        attack += self.b2b
                    self.b2b = -1
                
                self.update_b2b_level()

                if rows_cleared == 1:
                    if is_tspin == False:
                        attack += math.floor(0.5 + 0.25 * self.combo) ### formula exception
                    else:
                        if is_mini == True:
                            if self.b2b <= 0 and self.b2b_level <= 0:
                                attack += math.floor(0.5 + 0.25 * self.combo) ### formula exception
                            else:
                                attack += math.floor(min(max(self.b2b_level, 0), 1) * (1 + 0.25 * self.combo)) ### formula exception
                        else:
                            attack += math.floor(2 + min(max(self.b2b_level, 0), 1) * (1 + 0.25 * self.combo))
                
                else: attack += math.floor((1 + 0.25 * self.combo) * 
                                            (2 * rows_cleared * is_tspin * (-3/4 * is_mini + 1)
                                                + 2**(rows_cleared-2) * (1 - is_tspin)
                                                + min(max(self.b2b_level, 0), 1) * is_b2b)) # General formula for 2-4 rows cleared

                self.combo += 1

        self.lines_sent += attack
        self.lines_cleared += rows_cleared

        # Getting text for the type of attack
        attack_text = ''

        # If it's a mini the text will appear in the line above
        if is_mini == True:
            attack_text += 'MINI '

        if is_tspin == True:
            attack_text += 'T-SPIN '

        if rows_cleared == 4:
            attack_text += 'QUAD'
        elif rows_cleared == 3:
            attack_text += 'TRIPLE'
        elif rows_cleared == 2:
            attack_text +='DOUBLE'
        elif rows_cleared == 1:
            attack_text +='SINGLE'

        self.attack_text = attack_text

        return attack

    # def update_stats(self, rows_cleared, is_tspin, is_mini, is_all_clear):
    #     attack = self.get_attack()

    '''def update_stats(self, rows_cleared, is_tspin, is_mini, is_all_clear):
        attack = 0

        # doens't look very pretty now does it
        if rows_cleared == 0:
            self.combo = 0
        else:
            if rows_cleared == 1:
                if is_tspin == False:
                    self.b2b = -1
                    attack = math.floor(0.5 + 0.25 * self.combo) ###
                else:
                    self.b2b += 1
                    self.update_b2b_level()

                    if is_mini == True:
                        if self.b2b <= 0 and self.b2b_level <= 0:
                            attack = math.floor(0.5 + 0.25 * self.combo) ###
                        else:
                            attack = math.floor(self.b2b_level * (1 + 0.25 * self.combo)) ###
                    else:
                        attack = math.floor((2 + self.b2b_level) * (1 + 0.25 * self.combo))

            elif rows_cleared == 2:
                if is_tspin == False:
                    self.b2b = -1
                    attack = math.floor(1 + 0.25 * self.combo)
                else:
                    self.b2b += 1
                    self.update_b2b_level()

                    if is_mini == True:
                        attack = math.floor((1 + self.b2b_level) * (1 + 0.25 * self.combo))
                    else:
                        attack = math.floor((1 + 0.25 * self.b2b_level) * (4 + 1 * self.combo))

            elif rows_cleared == 3:
                if is_tspin == False:
                    self.b2b = -1
                    attack = math.floor(2 + 0.5 * self.combo)
                else:
                    self.b2b += 1
                    self.update_b2b_level()

                    attack = math.floor((1 + 1/6 * self.b2b_level) * (6 + 1.5 * self.combo))

            elif rows_cleared == 4:
                self.b2b += 1
                self.update_b2b_level()

                attack += math.floor((1 + 0.25 * self.b2b_level) * (4 + 1 * self.combo))

            self.combo += 1

        if is_all_clear:
            attack += 10

        self.lines_sent += attack

        self.lines_cleared += rows_cleared

        return attack'''

    # for making sure I implemented garbage sending correctly...
    # don't look please
    # it does work though

    def make_chart(self):
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
                chart[j][i].b2b_level = type[3]
                chart[j][i].get_attack(type[0],type[1],type[2], False)
                chart[j][i] = chart[j][i].lines_sent
        
        for row in chart:
            print(row)

if __name__ == "__main__":
    test = Stats()
    test.make_chart()