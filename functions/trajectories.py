import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from .gridworld import GridWorld, Agent


def random_policy(rng, probabilities):
    return rng.choice(
        [0, 1, 2, 3],
        p = probabilities
    )
def generate_trajectory(grid, policy, policy_probabilities, view_size, n_steps, move_seed, startx=None, starty=None, start_direction=None, random_start=False, start_seed=None):

    if random_start:
        startx, starty, start_direction = grid.randomXYStart(start_seed)
    
    elif startx is None or starty is None or start_direction is None:
        raise ValueError("startx, starty and direction must be provided when random start is false")
    
    agent = Agent(startx, starty, view_size, grid, start_direction)   #Create agent

    rng = np.random.default_rng(move_seed)

    for i in range(n_steps):                                  #Agent random walk
        action = policy(rng, policy_probabilities)
        observation = agent.action(action)

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
    print("NN inputs: ", inputs.shape)
    print("targets: ", target_obs.shape)
    
    return inputs, target_obs