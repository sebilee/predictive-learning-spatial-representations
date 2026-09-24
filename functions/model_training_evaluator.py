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


def model_training(model, training_input, training_output, evaluator_input, evaluator_output):

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
    for epoch in range(1000):
        optimizer.zero_grad()
        prediction = model(training_input)

        loss = loss_fn(prediction, training_output)
        loss_history.append(loss.item())
        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            #print(epoch, loss.item())
            trainer_error = model_evaluation(model, training_input, training_output, "trainer")
            validator_error = model_evaluation(model, evaluator_input, evaluator_output, "validator")
            trainer_error_log.append(trainer_error)
            validator_error_log.append(validator_error)
            epochs_recorded.append(epoch)
    #print(hidden_sequence.shape)
    print(prediction.shape)
    plt.plot(epochs_recorded, trainer_error_log, label = "training")
    plt.plot(epochs_recorded, validator_error_log, label = "test")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.legend()
    plt.show()

    return loss_history


def baseline_evaluation(inputs, targets):
    observation_size = targets.shape[-1]

    baseline_prediction = inputs[..., :observation_size]
    loss_fn = nn.MSELoss()
    baseline_loss = loss_fn(baseline_prediction, targets)

    print("Baseline loss: ", baseline_loss.item())



def model_evaluation(model, inputs, targets, name):

    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        prediction = model(inputs)
        test_loss = loss_fn(prediction, targets)
    print(test_loss.item())

    print(f"Test loss, {name}: ", test_loss.item())
    model.train()
    return test_loss.item()