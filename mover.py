"""Handles DAS, ARR, and Softdrop for the Human Player"""
import time
from const import *

class Mover:
    def __init__(self):
        # Separating lr and sd das
        self.can_lr_das = True # True if DAS not into a wall
        self.lr_das_start_time = None # Used for DAS and ARR
        self.lr_das_direction = None # DAS direction
        self.lr_das_counter = 0 # Used for ARR timing

        self.right_held = False
        self.left_held = False

        # Soft drop timing
        self.can_sd_das = True
        self.sd_start_time = None
        self.sd_counter = 0
        self.sd_held = False
    
    # DAS and ARR methods
    def start_left(self):
        self.lr_das_start_time = time.time()
        self.lr_das_direction = "L"

        self.lr_das_counter = DAS/1000
        self.left_held = True
        
        self.can_lr_das = True

    def start_right(self):
        self.lr_das_start_time = time.time()
        self.lr_das_direction = "R"

        self.lr_das_counter = DAS/1000
        self.right_held = True

        self.can_lr_das = True

    def stop_left(self): # pain
        self.left_held = False
        if (self.lr_das_direction == "L"):
            if (self.right_held == False):
                self.lr_das_start_time = None
                self.lr_das_counter = DAS/1000
            else: # Still holding other key
                self.can_lr_das = True
                self.lr_das_start_time = time.time()
                self.lr_das_direction = "R"
                self.lr_das_counter = DAS/1000

    def stop_right(self): # also pain
        self.right_held = False
        if (self.lr_das_direction == "R"):
            if (self.left_held == False):
                self.lr_das_start_time = None
                self.lr_das_counter = DAS/1000
            else: # Still holding other key
                self.can_lr_das = True
                self.lr_das_start_time = time.time()
                self.lr_das_direction = "L"
                self.lr_das_counter = DAS/1000

    # Soft drop methods

    def start_down(self):
        self.sd_start_time = time.time()
        self.sd_movement_conunter = 0

        self.sd_held = True
    
    def stop_down(self):
        self.sd_start_time = None
        self.sd_counter = 0

        self.sd_held = False