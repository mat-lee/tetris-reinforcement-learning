"""Handles DAS, ARR, and Softdrop"""
import time
from const import *

class Mover:
    def __init__(self):
        self.das_start_time = None
        self.das_direction = None
        self.counter = 0

        # Prevent code from trying to move the piece forever
        self.movement_counter = 0
        self.movement_counter_max = COLS - 1
        self.right_held = False
        self.left_held = False

        # Soft drop timing
        self.sd_start_time = None
        self.sd_counter = 0

        # Prevent code from trying to softdrop the piece forever
        self.sd_movement_conunter = 0
        self.sd_movement_counter_max = ROWS - 1

        self.movement_string = ""
    
    # DAS and ARR methods

    def reset_counter(self, is_sd=False):
        if is_sd:
            self.movement_counter = 0
        else:
            self.sd_movement_conunter = 0

    def start_left(self):
        self.movement_string += "L"
        self.das_start_time = time.time()
        self.das_direction = "left"

        self.movement_counter = 0
        self.counter = DAS/1000
        self.left_held = True

    def start_right(self):
        self.movement_string += "R"
        self.das_start_time = time.time()
        self.das_direction = "right"

        self.movement_counter = 0
        self.counter = DAS/1000
        self.right_held = True

    def stop_left(self): # pain
        self.left_held = False
        if (self.das_direction == "left"):
            if (self.right_held == False):
                self.das_start_time = None
                self.counter = DAS/1000
            else:
                self.das_start_time = time.time()
                self.das_direction = "right"
                self.counter = DAS/1000

    def stop_right(self): # also pain
        self.right_held = False
        if (self.das_direction == "right"):
            if (self.left_held == False):
                self.das_start_time = None
                self.counter = DAS/1000
            else:
                self.das_start_time = time.time()
                self.das_direction = "left"
                self.counter = DAS/1000

    # Soft drop methods

    def start_down(self):
        self.sd_start_time = time.time()
        self.sd_movement_conunter = 0
    
    def stop_down(self):
        self.sd_start_time = None
        self.sd_counter = 0
    
    # Movement queue methods

    def update_queue(self, time, is_sd=False):
        # Left-right queue
        if not is_sd:
            if self.das_start_time != None:
                if self.movement_counter < self.movement_counter_max:
                    if time - self.das_start_time > self.counter:
                        if self.das_direction == "left":
                            self.movement_string += "L"
                        elif self.das_direction == "right":
                            self.movement_string += "R"
                        self.counter += ARR/1000
                        self.movement_counter += 1
        else:
            # Softdrop queue
            if self.sd_start_time != None:
                if self.sd_movement_conunter < self.sd_movement_counter_max:
                    if time - self.sd_start_time > self.sd_counter:
                        self.movement_string += "D"
                        self.sd_counter += (1 / SDF) / 1000
                        self.sd_movement_conunter += 1