import random
from math import floor
import scriptcontext as sc
import ghpythonlib.components as ghcomp
import ghpythonlib.treehelpers as th


class BeadyRing_env:
    def __init__(self):
        self._carrier_color = 127.5
        self._house_color = 0
        self._street_color = 255
        self._cell_size = 3
        self._max_row_len = 41
        self._3d = False
        self.iter = 0
        
        # grid world
        xy_plane = ghcomp.XYPlane(ghcomp.ConstructPoint(0, 0, 0))
        cell, _ = ghcomp.Rectangle(xy_plane, self._cell_size, self._cell_size, 0)
        
        r_max = self._cell_size * self._max_row_len
        move_range = [i for i in range(0, r_max, self._cell_size)]
        
        y_vec = ghcomp.UnitY(move_range)
        cell_col, _ = ghcomp.Move(cell, y_vec)
        
        x_vec = ghcomp.UnitX(move_range)
        grid_world = []
        for c in cell_col:
            cell_row, _ = ghcomp.Move(c, x_vec)
            grid_world.append(cell_row)
        
        self.grid_world = ghcomp.ReverseList(grid_world)
        
        # initial location
        self.R = int(floor(self._max_row_len/2))
        self.C = int(floor(self._max_row_len/2))
        
        R_space = [r for r in range(0, self._max_row_len)]
        C_space = [c for c in range(0, self._max_row_len)]
        self.RC_space = [[[r, c] for c in C_space] for r in R_space]
        
        self.cell = [self.R, self.C]
        self.adjacent_cells = [[self.R, self.C]]
        self.adj_cells = [[self.R, self.C]]
        
        # initial state
        self.state = [[self._carrier_color for _ in range(self._max_row_len)] 
                        for _ in range(self._max_row_len)]

    def step(self, _cell_state):
        self.iter += 1
        # update state step
        self.state[self.R][self.C] = 255*_cell_state
        
        for i, a in enumerate(self.adj_cells):
            if a == self.cell:
                del self.adj_cells[i]
        
        adjacent = self.get_adjacent()
        
        # reward
        reward = 0
        adj_street_count = 0
        
        for a in adjacent:
            if int(self.state[a[0]][a[1]]) == 255:
                adj_street_count += 1
        
        if _cell_state == 1:
            if adj_street_count >= 3:
                reward -= 1
            elif adj_street_count == 0:
                reward -= 1
            elif 0 < adj_street_count < 3:
                reward += 1
        elif _cell_state == 0:
            # density metric
            reward += 1/(self._max_row_len**2)
            if adj_street_count == 0:
                reward -= 1
            elif adj_street_count >= 3:
                reward -= 1
            elif 0 < adj_street_count < 3:
                reward += 2
        
        for item in adjacent:
            if item not in self.adjacent_cells:
                self.adjacent_cells.append(item)
                self.adj_cells.append(item)
        
        # done
        done = False
        if len(self.adj_cells) == 0:
            done = True
        elif len(self.adj_cells) > 0:
            # next action location selection
            self.cell = random.choice(self.adj_cells)
            self.R = self.cell[0]
            self.C = self.cell[1]
        
        return self.state, reward, done, {}

    def reset(self):
        # initial state
        self.state = [[self._carrier_color for _ in range(self._max_row_len)] 
                        for _ in range(self._max_row_len)]
        
        # reset step counter
        self.iter = 0
        
        # initial location
        self.R = int(floor(self._max_row_len/2))
        self.C = int(floor(self._max_row_len/2))
        
        self.cell = [self.R, self.C]
        self.adjacent_cells = [[self.R, self.C]]
        self.adj_cells = [[self.R, self.C]]
        
        return self.state

    def render(self):
        # state visualization
        color_state = [ghcomp.ColourRGB(255, self.state[i], self.state[i], 
                        self.state[i]) for i in range(self._max_row_len)]
        return color_state

    def get_adjacent(self):
        adjacent = []
        if self.R < len(self.RC_space) - 1:
            adjacent.append(self.RC_space[self.R+1][self.C])
        if self.C > 0:
            adjacent.append(self.RC_space[self.R][self.C-1])
        if self.R > 0:
            adjacent.append(self.RC_space[self.R-1][self.C])
        if self.C < len(self.RC_space[self.R]) - 1:
            adjacent.append(self.RC_space[self.R][self.C+1])
        return adjacent

    def get_house_cells(self):
        ## house cells
        house_cells_ = []
        for i in range(self._max_row_len):
            for j in range(self._max_row_len):
                if self.state[i][j] == 0:
                    house_cells_.append(self.grid_world[i][j])
        return house_cells_

if 'env' not in globals():
    env = BeadyRing_env()
    sc.sticky['env'] = env

if reset:
    state = sc.sticky['env'].reset()
    reward = None
    done = False
    info = {}
    
elif action is not None:
    state, reward, done, info = sc.sticky['env'].step(action)

else:
    raise RuntimeError("Either reset or action must be provided")

if render:
    grid_world_ = th.list_to_tree(sc.sticky['env'].grid_world, source=[0,0])
    color_state = sc.sticky['env'].render()
    color_state_ = th.list_to_tree(color_state, source=[0,0])
    if sc.sticky['env']._3d:
        house_cells = sc.sticky['env'].get_house_cells()
        house_cells_ = th.list_to_tree(house_cells, source=[0,0])

sc.sticky["state"] = state
sc.sticky["reward"] = reward
sc.sticky["done"] = done
sc.sticky["info"] = info