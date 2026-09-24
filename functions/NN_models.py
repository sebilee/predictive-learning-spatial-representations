import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from .gridworld import GridWorld, Agent


class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, nonlinearity):
        super().__init__()

        self.hidden = nn.Linear(input_size, hidden_size)

        if nonlinearity == "relu":
            self.activation == nn.ReLU()
        elif nonlinearity == "tanh":
            self.activation == nn.tanh()
        else:
            raise ValueError("Only ReLU and Tanh activations are allowed")

        self.decoder = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h = self.activation(self.hidden(x))
            prediction = self.decoder(h)
            return prediction

        def get_hidden_states(self, x):
            h = self.activation(self.hidden(x))
            return h




class BasicRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, nonlinearity):
        super().__init__()

        self.rnn = nn.RNN(
            input_size = input_size,
            hidden_size = hidden_size,
            nonlinearity = nonlinearity,
            batch_first = True
        )

        self.decoder = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):
        hidden_sequence, final_hidden = self.rnn(x)

        prediction = self.decoder(hidden_sequence)
        return prediction

    def get_hidden_states(self, x):
        hidden_sequence, _ = self.rnn(x)

        return hidden_sequence