import torch
from .trajectories import generate_trajectory, prepare_data


def generate_dataset(grid, policy, action_probs, view_size, n_steps, traj_count, move_seed_start = 0, startx=None, starty=None, start_direction=None, random_start=False, start_seed_start=10000):


    agent_list = []
    inputs = []
    targets = []

    for i in range(traj_count):
        move_seed = move_seed_start + i
        start_seed = start_seed_start + i
        agent = generate_trajectory(grid, policy, action_probs, view_size, n_steps, move_seed, startx=startx, starty=starty, start_direction=start_direction, random_start=random_start, start_seed=start_seed)
        agent_list.append(agent)

        input_data, target = prepare_data(agent)
        inputs.append(input_data)
        targets.append(target)

    inputs_stack = torch.stack(inputs, dim = 0)
    targets_stack = torch.stack(targets, dim = 0)

    return inputs_stack, targets_stack, agent_list