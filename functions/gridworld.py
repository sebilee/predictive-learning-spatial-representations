import sys
from pathlib import Path

ROOT = Path.cwd().parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

class Agent:
    def __init__(self, startx, starty, viewsize, gridworld, direction = 0):
        self.gridworld = gridworld
        self.position = [startx, starty]
        self.direction = direction
        self.view = np.zeros((viewsize, viewsize))
        self.viewsize = viewsize
        self.observations = []
        self.observe()
        self.states = [[startx, starty, direction]]    
        self.actions = []

    def rotate(self, rotation):
        xpos, ypos = self.position
        direction = self.direction
        if rotation == 1:   #left rotation
            self.actions.append(1)
            if direction > 0:
                direction -= 1
            else:
                direction = 3
        if rotation == 2:   #right rotation
            self.actions.append(2)
            if direction < 3:
                direction += 1
            else:
                direction = 0
        self.direction = direction
        self.states.append([xpos, ypos, direction])

    def is_walkable(self, x, y):
        return self.gridworld.grid[x, y] != -1

    def action(self, action):
        if action == 0:
            self.move()
        elif action == 1:
            self.rotate(1)
        elif action == 2:
            self.rotate(2)
        else:
            self.stop()
        return self.observe()
        


    def move(self):
        xpos, ypos = self.position
        direction = self.direction
        if direction == 0:  #north
            if self.is_walkable(xpos - 1, ypos):
                xpos -= 1
        elif direction == 1:    #east
            if self.is_walkable(xpos, ypos + 1):
                ypos += 1
        elif direction == 2:    #south
            if self.is_walkable(xpos + 1, ypos):
                xpos += 1
        elif direction == 3:    #west
            if self.is_walkable(xpos, ypos - 1):
                ypos -= 1

        self.position = [xpos, ypos]
        self.states.append([xpos, ypos, direction])
        self.actions.append(0)


    def stop(self):
        xpos, ypos = self.position
        direction = self.direction

        self.states.append([xpos, ypos, direction])
        self.actions.append(3)

    def observe(self):
        grid = self.gridworld
        xpos, ypos = self.position
        xview, yview = self.position
        direction = self.direction
        viewsize = self.viewsize
        half = int(viewsize / 2)
        if direction == 0:
            xview = xpos - half
        elif direction == 1:
            yview = ypos + half
        elif direction == 2:
            xview = xpos + half
        elif direction == 3:
            yview = ypos - half
        x0 = xview - half
        x1 = xview + half
        y0 = yview - half
        y1 = yview + half
        patch = grid.grid[x0:x1 + 1, y0:y1+1].copy()
        if direction == 1:
            self.view = np.rot90(patch, 1)
        elif direction == 2:
            self.view = np.rot90(patch, 2)
        elif direction == 3:
            self.view = np.rot90(patch, 3)
        else:
            self.view = patch
        self.observations.append(self.view.copy())
        #print("observed")
        return self.view.copy()
            
        

class GridWorld:
    def __init__(self, height, width):
        self.grid = np.full((height, width), -1)

    def _update_valid_positions(self):
        self.valid_positions = np.argwhere(np.array(self.grid) != -1)


    def as_array(self):
        return self.grid.copy()

    def add_rectangle_region(self, row1, row2, col1, col2):
        self.grid[row1:row2, col1:col2] = 0
        self._update_valid_positions()

    def add_single_shape(self, x, y, color):
        self.grid[x, y] = color
        self._update_valid_positions()

    def add_square_shape(self, x, y, size, color):
        self.grid[x:x+size, y:y+size] = color
        self._update_valid_positions()

    def add_rectangle_shape(self, x, y, height, width, color):
        self.grid[x:x+height, y:y+width] = color
        self._update_valid_positions()

    def add_triangle_shape(self, x, y, size, orientation, color):
        if orientation == 0: #bottom right sides
            for i in range(size):
                self.grid[x+i:x+size, y + i] = color
        if orientation == 1: # top right sides
            for i in range(size):
                self.grid[x+size-i:x+size, y + i] = color 
        if orientation == 2: #top left sides
            for i in range(size):
                self.grid[x:x+i, y + i] = color
        if orientation == 3: #bottom left sides
            for i in range(size):
                self.grid[x:x+size-i, y + i] = color
        self._update_valid_positions()

    def randomXYStart(self, seed):

        valid_positions = self.valid_positions

        rng = np.random.default_rng(seed)

        x, y = valid_positions[rng.integers(len(valid_positions))]
        direction = rng.integers(4)

        return x, y, direction
            