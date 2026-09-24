import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

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

    def as_array(self):
        return self.grid.copy()

    def add_rectangle_region(self, row1, row2, col1, col2):
        self.grid[row1:row2, col1:col2] = 0

    def add_single_shape(self, x, y, color):
        self.grid[x, y] = color

    def add_square_shape(self, x, y, size, color):
        self.grid[x:x+size, y:y+size] = color

    def add_rectangle_shape(self, x, y, height, width, color):
        self.grid[x:x+height, y:y+width] = color

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
        

HEIGHT = 20
WIDTH = 25
VIEW_SIZE = 5

START_X = 6
START_Y = 6
START_DIRECTION = 2

N_STEPS = 500

ACTION_PROBS = [0.6, 0.15, 0.15, 0.1]

SEED = 42


grid = GridWorld(HEIGHT, WIDTH)
grid.add_rectangle_region(5, 10, 5, 20)
grid.add_rectangle_region(10, 15, 10, 15)
grid.add_square_shape(6, 6, 3, 7)
grid.add_triangle_shape(6, 11, 3, 0, 6)
grid.add_single_shape(7, 17, 5)
grid.add_single_shape(6, 16, 5)
grid.add_single_shape(6, 18, 5)
grid.add_single_shape(8, 16, 5)
grid.add_single_shape(8, 18, 5)
grid.add_rectangle_shape(11, 12, 3, 1, 3)
grid.add_rectangle_shape(12, 11, 1, 3, 4)


def generate_trajectory(grid, start_x, start_y, start_direction, view_size, n_steps, seed):
    
    agent = Agent(start_x, start_y, view_size, grid, start_direction)   #Create agent

    rng = np.random.default_rng(seed)

    for i in range(n_steps):                                            #Agent random walk
        action = rng.choice([0, 1, 2, 3], p=[0.6, 0.15, 0.15, 0.1])

        if action == 0:
            agent.move()
            #print(f"moved towards {agent.direction}")
        elif action == 3:
            agent.stop()
            #print("stopped")
        else:

            agent.rotate(action)
            #print(f"rotated towards {agent.direction}")
        observation = agent.observe()
        #print(pd.DataFrame(observation))

    #print(agent.states)
    print(len(agent.states))
    print(len(agent.observations))
    print(len(agent.actions))

    return agent


def prepare_data(agent):                                                                #Conversion to tensors
    observations = torch.tensor(
        np.array(agent.observations),
        dtype = torch.float32
    )
    actions = torch.tensor(
        agent.actions, 
        dtype = torch.long
    )
    states = torch.tensor(
        agent.states,
        dtype = torch.long
    )
    print(observations.shape)
    print(actions.shape)
    print(states.shape)
    observations = observations.flatten(start_dim = 1)
    print(observations.shape)
    actions_onehot = torch.nn.functional.one_hot(
        actions,
        num_classes = 4
    ).float()
    print(actions_onehot.shape)
    input_obs = observations[:-1]
    target_obs = observations[1:]
    inputs = torch.cat(
        (input_obs, actions_onehot),
        dim = 1
    )
    print("input observations: ", input_obs.shape)
    print("actions: ", actions_onehot.shape)
    print("RNN inputs: ", inputs.shape)
    print("targets: ", target_obs.shape)
    
    return inputs, target_obs



def model_creation(input, output, hidden_size):

    model = nn.Sequential(
        nn.Linear(trainer_data_inputs.shape[1], hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output.shape[1])
    )
    return model

def model_evaluation(model, observation, inputs, name):

    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        prediction = model(inputs)
        test_loss = loss_fn(prediction, observation)
    print(test_loss.item())

    print(f"Test loss, {name}: ", test_loss.item())

    return test_loss.item()

def baseline_evaluation(inputs, targets):
    observation_size = targets.shape[1]

    baseline_prediction = inputs[:, :observation_size]
    loss_fn = nn.MSELoss()
    baseline_loss = loss_fn(baseline_prediction, targets)

    print("Baseline loss: ", baseline_loss.item())


def model_training(model, input, output):

    loss_fn = nn.MSELoss()

    loss_history = []

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    model.train()

    trainer_error_log = []
    tester_error_log = []
    epochs_recorded = []
    for epoch in range(3000):
        optimizer.zero_grad()
        prediction = model(input)

        loss = loss_fn(prediction, output)
        loss_history.append(loss.item())
        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            #print(epoch, loss.item())
            trainer_error = model_evaluation(model_NN, trainer_data_target_obs, trainer_data_inputs, "trainer")
            tester_error = model_evaluation(model_NN, tester_data_target_obs, tester_data_inputs, "tester")
            trainer_error_log.append(trainer_error)
            tester_error_log.append(tester_error)
            epochs_recorded.append(epoch)

    plt.plot(epochs_recorded, trainer_error_log, label = "training")
    plt.plot(epochs_recorded, tester_error_log, label = "test")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.legend()
    plt.savefig("plot.png")

    return loss_history
    
   
trainer_agent = generate_trajectory(grid, START_X, START_Y, START_DIRECTION, VIEW_SIZE, N_STEPS, SEED)
trainer_data_inputs, trainer_data_target_obs = prepare_data(trainer_agent)

TEST_SEED = 123
TEST_STEPS = 200

tester_agent = generate_trajectory(grid, START_X, START_Y, START_DIRECTION, VIEW_SIZE, TEST_STEPS, TEST_SEED)
tester_data_inputs, tester_data_target_obs = prepare_data(tester_agent)

INPUT_SIZE = trainer_data_inputs.shape[1]
OUTPUT_SIZE = trainer_data_target_obs.shape[1]
HIDDEN_SIZE = 64     


model_NN = model_creation(trainer_data_inputs, trainer_data_target_obs, HIDDEN_SIZE)
loss_history = model_training(model_NN, trainer_data_inputs, trainer_data_target_obs)


model_evaluation(model_NN, trainer_data_target_obs, trainer_data_inputs, "trainer")
model_evaluation(model_NN, tester_data_target_obs, tester_data_inputs, "tester")
baseline_evaluation(tester_data_inputs, tester_data_target_obs)

