import torch  # Import the core PyTorch library so we can create tensors, models, and training utilities.
from torch import nn  # Import the neural network module namespace for layers like Linear and ReLU.
from torch.utils.data import DataLoader  # Import the data loader to batch and iterate over datasets.
from torchvision import datasets  # Import torchvision datasets for loading FashionMNIST.
from torchvision.transforms import v2  # Import the modern torchvision transform API.

training_data = datasets.FashionMNIST(  # Create the training dataset for FashionMNIST.
    root="data",  # Store the dataset in the local "data" folder.
    train=True,  # Use the training split of the dataset.
    download=True,  # Download the dataset if it is not already present locally.
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),  # Convert images to tensors in float32 format and scale pixel values to [0, 1].
)

test_data = datasets.FashionMNIST(  # Create the test dataset for evaluation.
    root="data",  # Use the same local data directory as the training set.
    train=False,  # Use the test split of the dataset.
    download=True,  # Download the dataset if needed.
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),  # Convert test images to float32 tensors scaled to [0, 1].
)

batch_size = 64  # Set how many samples are processed together in each training/test batch.

train_dataloader = DataLoader(training_data, batch_size=batch_size)  # Wrap the training data in a loader that yields batches of size 64.
test_dataloader = DataLoader(test_data, batch_size=batch_size)  # Wrap the test data in a loader that yields batches of size 64.

for X, y in test_dataloader:  # Loop through the first batch of the test data to inspect its shape.
    print(f"Shape of X [N, C, H, W]: {X.shape}")  # Print the batch tensor shape: (batch_size, channels, height, width).
    print(f"Shape of y: {y.shape} {y.dtype}")  # Print the corresponding label tensor shape and data type.
    break  # Stop after the first batch so the output stays short.

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # Choose the best available accelerator if present; otherwise use CPU.
print(f"Using {device} device")  # Print which device the model will run on.

class NeuralNetwork(nn.Module):  # Define a custom neural network by inheriting from PyTorch's nn.Module base class.
    def __init__(self):  # Initialize the network layers.
        super().__init__()  # Call the parent nn.Module initializer so the module is set up correctly.
        self.flatten = nn.Flatten()  # Create a layer that flattens each image from 28x28 into a 784-length vector.
        self.linear_relu_stack = nn.Sequential(  # Build a sequence of layers for the network.
            nn.Linear(28*28, 512),  # Fully connected layer that maps 784 inputs to 512 hidden units.
            nn.ReLU(),  # Apply the ReLU activation function to introduce non-linearity.
            nn.Linear(512, 512),  # Second hidden layer with 512 inputs and 512 outputs.
            nn.ReLU(),  # Apply ReLU again after the second linear layer.
            nn.Linear(512, 10)  # Final output layer with 10 logits, one for each class.
        )

    def forward(self, x):  # Define how input data flows through the network.
        x = self.flatten(x)  # Flatten the input image tensor from [N, 1, 28, 28] to [N, 784].
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)  # Instantiate the model and move it to the selected device (CPU or GPU).
print(model)

loss_fn = nn.CrossEntropyLoss()  # Use cross-entropy loss, which is suitable for multi-class classification problems.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # Use stochastic gradient descent with a learning rate of 0.001 to optimize the model parameters.

def train(dataloader, model, loss_fn, optimizer):  # Define a function to train the model for one epoch.
    size = len(dataloader.dataset)  # Get the total number of samples in the dataset.
    model.train()  # Set the model to training mode, enabling features like dropout and batch normalization.
    for batch, (X, y) in enumerate(dataloader):  # Loop through each batch of data.
        X, y = X.to(device), y.to(device)  # Move the input data and labels to the selected device.

        pred = model(X)  # Perform a forward pass through the model to get predictions.
        loss = loss_fn(pred, y)  # Compute the loss between predictions and true labels.

        loss.backward()  # Backpropagate the loss to compute gradients for each parameter.
        optimizer.step()  # Update the model parameters based on the computed gradients.
        optimizer.zero_grad()  # Reset the gradients to zero before the next iteration.

        if batch % 100 == 0:  # Print progress every 100 batches.
            loss, current = loss.item(), (batch + 1) * len(X)  # Get the current loss value and the number of samples processed so far.
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]") # Print the loss and progress in a formatted string.

def test(dataloader, model, loss_fn):  # Define a function to evaluate the model on the test dataset.
    size = len(dataloader.dataset)  # Get the total number of samples in the test dataset.
    num_batches = len(dataloader)  # Get the number of batches in the test dataloader.
    model.eval()  # Set the model to evaluation mode, disabling features like dropout.
    test_loss, correct = 0, 0  # Initialize variables to accumulate loss and count correct predictions.
    with torch.no_grad():  # Disable gradient computation for efficiency during evaluation.
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # Move the input data and labels to the selected device.
            pred = model(X)  # Perform a forward pass to get predictions.
            test_loss += loss_fn(pred, y).item()  # Accumulate the loss for this batch.
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # Count the number of correct predictions in this batch.
    test_loss /= num_batches # Compute the average loss over all batches.
    correct /= size  # Compute the accuracy as the ratio of correct predictions to total samples.
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5  # Set the number of epochs to train the model.
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")  # Print the current epoch number.
    train(train_dataloader, model, loss_fn, optimizer)  # Call the training function for one epoch.
    test(test_dataloader, model, loss_fn)  # Evaluate the model on the test dataset after training.
print("Done!")  # Indicate that training and evaluation are complete.

torch.save(model.state_dict(), "model.pth")  # Save the trained model's parameters to a file named "model.pth".
print("Saved PyTorch Model State to model.pth")