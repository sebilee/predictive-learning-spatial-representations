import sys
from pathlib import Path
ROOT = Path.cwd().parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt


def model_training(model, training_input, training_output, evaluator_input, evaluator_output, epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    training_input, training_output, evaluator_input, evaluator_output = [
        x.to(device)
        for x in (training_input, training_output, evaluator_input, evaluator_output)
        ]


    loss_fn = nn.MSELoss()

    loss_history = []

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    model.train()

    trainer_error_log = []
    validator_error_log = []
    epochs_recorded = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction = model(training_input)

        loss = loss_fn(prediction, training_output)
        loss_history.append(loss.item())
        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            #print(epoch)
            trainer_error = model_evaluation(model, training_input, training_output)
            validator_error = model_evaluation(model, evaluator_input, evaluator_output)
            trainer_error_log.append(trainer_error)
            validator_error_log.append(validator_error)
            epochs_recorded.append(epoch)
    #print(f"Trainer error: {trainer_error}")
    #print(f"Validator error: {validator_error}")
    #print(hidden_sequence.shape)
    #print(prediction.shape)
    loss_info = []
    loss_info.append(epochs_recorded)
    loss_info.append(trainer_error_log)
    loss_info.append(validator_error_log)
    return loss_info

def plot_training_result(loss_info, xlabel, ylabel, name_trainer, name_validator):
    plt.plot(loss_info[0], loss_info[1], label = name_trainer)
    plt.plot(loss_info[0], loss_info[2], label = name_validator)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    


def baseline_evaluation(inputs, targets):
    observation_size = targets.shape[-1]

    baseline_prediction = inputs[..., :observation_size]
    loss_fn = nn.MSELoss()
    baseline_loss = loss_fn(baseline_prediction, targets)

    return baseline_loss.item()



def model_evaluation(model, inputs, targets):
    
    inputs, targets = to_model_device(model, inputs, targets)

    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        prediction = model(inputs)
        test_loss = loss_fn(prediction, targets)
    #print(test_loss.item())

    #print(f"Test loss, {name}: ", test_loss.item())
    model.train()
    return test_loss.item()


def model_evaluation_memoryless(model, inputs, targets):

    inputs, targets = to_model_device(model, inputs, targets)

    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        prediction = model.forward_no_memory(inputs)
        test_loss = loss_fn(prediction, targets)
    #print(test_loss.item())

    #print(f"Test loss, {name}: ", test_loss.item())
    model.train()
    return test_loss.item()

def evaluator_per_action(model, inputs, targets):

    inputs, targets = to_model_device(model, inputs, targets)

    loss_fn = nn.MSELoss()
    actions_onehot = inputs[..., -4:]
    actions = actions_onehot.argmax(dim=-1)
    action_losses = [None, None, None, None]
    model.eval()
    prediction = model(inputs)
    for i in range(4):
        mask = actions == i
        if mask.sum() == 0:
            print(f"No samples for {i}")
            continue
        movement_predictions = prediction[mask]
        movement_targets = targets[mask]
        loss = loss_fn(movement_predictions, movement_targets)
        action_losses[i] = loss.item()

    #print(f"{name} loss per action: ")
    #print(f"Forward: {action_losses[0]}")
    #print(f"Left: {action_losses[1]}")
    #print(f"Right: {action_losses[2]}")
    #print(f"Stop: {action_losses[3]}")

    return action_losses

def evaluator_per_action_memoryless(model, inputs, targets):

    inputs, targets = to_model_device(model, inputs, targets)
    loss_fn = nn.MSELoss()
    actions_onehot = inputs[..., -4:]
    actions = actions_onehot.argmax(dim=-1)
    action_losses = [None, None, None, None]
    model.eval()
    prediction = model.forward_no_memory(inputs)
    for i in range(4):
        mask = actions == i
        if mask.sum() == 0:
            print(f"No samples for {i}")
            continue
        movement_predictions = prediction[mask]
        movement_targets = targets[mask]
        loss = loss_fn(movement_predictions, movement_targets)
        action_losses[i] = loss.item()

    #print(f"{name} loss per action: ")
    #print(f"Forward: {action_losses[0]}")
    #print(f"Left: {action_losses[1]}")
    #print(f"Right: {action_losses[2]}")
    #print(f"Stop: {action_losses[3]}")

    return action_losses

def to_model_device(model, *tensors):
    device = next(model.parameters()).device
    return tuple(t.to(device) for t in tensors)