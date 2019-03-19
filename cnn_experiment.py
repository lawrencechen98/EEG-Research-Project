import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import numpy as np


import matplotlib.pyplot as plt
from tqdm import tqdm

from math import ceil

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def get_train_valid_split(num_valid=415): 
    
    person_train_valid = np.load("data/person_train_valid.npy")
    X_train_valid = np.load("data/X_train_valid.npy")
    y_train_valid = np.load("data/y_train_valid.npy")
    
    # Preprocess labels
    label_translations = {769:0, 770:1, 771:2, 772:3}
    for label in label_translations:
        y_train_valid[y_train_valid==label] = label_translations[label]
    
    
    num_trials, _ = person_train_valid.shape
    
    # Choose Validation Subset Randomly
    valid_indices = np.random.choice(np.arange(num_trials), size=num_valid, replace=False)
    valid_mask = np.zeros(num_trials, dtype=bool)
    valid_mask[valid_indices] = True
    train_mask = ~valid_mask  
    
    person_train, X_train, y_train = person_train_valid[train_mask, :], \
        X_train_valid[train_mask, :, :], \
        y_train_valid[train_mask]
    
    person_valid, X_valid, y_valid = person_train_valid[valid_mask, :], \
        X_train_valid[valid_mask, :, :], \
        y_train_valid[valid_mask]
    
    return person_train, X_train, y_train, person_valid, X_valid, y_valid

def get_test_data():
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    person_test = np.load("data/person_test.npy")
    
    # Preprocess labels
    label_translations = {769:0, 770:1, 771:2, 772:3}
    for label in label_translations:
        y_test[y_test==label] = label_translations[label]
        
    return person_test, X_test, y_test
    

def get_train_loader(X_train, y_train, batch_size=50, shuffle=False):
    if shuffle == True:
        raise NotImplementedError
        
    num_batches = ceil(X_train.shape[0] / batch_size)
    
    # Lambda returns a reusable closure that returns a new generator
    # (HAXXXX)
    return num_batches, lambda: ((X_train[i*num_batches : (i+1)*num_batches, :, :],
                         y_train[i*num_batches : (i+1)*num_batches])
                         for i in range(num_batches))


def get_accuracy_from_loader(model, data_loader):

    ## ""Portability""
    num_correct = 0. # I have no idea why I'm doing this
    num_total = 0. # I'm pretty sure the code'll break with Python 2, anyway

    with torch.no_grad():
        for trials, labels in data_loader():
            trials, labels = torch.tensor(trials, dtype=torch.float, device=device), \
                torch.tensor(labels, device=device, dtype=torch.long)
            predictions = model(trials).argmax(dim=1)

            num_total += labels.numel()
            num_correct += predictions.eq(labels).sum().item()

    return num_correct / num_total

def get_accuracy_from_data(model, trials, labels):

    with torch.no_grad():
        predictions = model(trials).argmax(dim=1)
        num_total = labels.numel()
        num_correct = predictions.eq(labels).sum().item()

    return num_correct * 1.0 / num_total


def plot_losses(losses, modelname='1-D CNN', show=True, filename=None):
    plt.figure()
    plt.plot(np.arange(len(losses), dtype=int)+1, losses)
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.title(modelname+' Loss history')
    plt.gcf().set_size_inches(10, 8)
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

def plot_accuracies(train_acc, val_acc, modelname='1-D CNN', show=True, filename=None):
    plt.figure()
    plt.plot(np.arange(len(train_acc), dtype=int)+1, train_acc, label='train')
    plt.plot(np.arange(len(val_acc), dtype=int)+1, val_acc, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracies')
    plt.title(modelname+' Accuracy History')
    plt.legend()
    plt.gcf().set_size_inches(10, 8)
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()


class OneDimCNN(nn.Module):
    def __init__(self):
        super(OneDimCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(13, 1), stride=(1, 1), padding=(6, 0)),
            nn.AvgPool2d(kernel_size=(48, 1), stride=(8, 1)),
            nn.ELU(),
            # nn.BatchNorm2d(50),
            nn.Dropout2d(),
            # After above CxHxW = 50x120x1

            nn.Conv2d(50, 35, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.ELU(),
            # nn.BatchNorm2d(35),
            nn.Dropout2d()
            # After above CxHxW = 35x60x1
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(35*60, 400),
            nn.BatchNorm1d(400),
            nn.Dropout(),
            nn.ELU(),

            nn.Linear(400, 4)
        )

    def forward(self, x):
        x_extra_dim = x[:, :, :, None]
        conv_out = self.conv_layers(x_extra_dim)
        flat_conv_out = conv_out.reshape(x.size(0), -1)
        out = self.fc_layers(flat_conv_out)
        return out

# Get Data and Loaders
person_train, X_train, y_train, person_valid, X_valid, y_valid = get_train_valid_split()
person_test, X_test, y_test = get_test_data()

num_batches, train_loader = get_train_loader(X_train, y_train)


# Instantiate Model
model = OneDimCNN().to(device=device)

# Set Optimizer and Loss Criterion
lr = 2e-4
lr_decay = 0.6
epochs_per_decay = 20
weight_decay = 1e-2

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()


# Training
num_epochs = 100
epoch_losses = []
epoch_val_acc = []
epoch_train_acc = []

# Move Validation to GPU for speed
X_valid_tensor, y_valid_tensor = torch.tensor(X_valid, device=device, dtype=torch.float), \
    torch.tensor(y_valid, dtype=torch.long, device=device)

for epoch in range(num_epochs):
    batch_losses = []
    batch_accuracies = []
    for trials, labels in tqdm(train_loader(), total=num_batches):
        trials, labels = torch.tensor(trials, device=device, dtype=torch.float), \
            torch.tensor(labels, device=device, dtype=torch.long)

        # Compute Loss
        scores = model(trials)
        loss = criterion(scores, labels)

        # Backprop + Optim Step
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Add Losses to batch losses for the PLOTS
        batch_losses.append(loss.item())
        batch_accuracies.append(get_accuracy_from_data(model, trials, labels))

    # calculate epoch accuracy and loss
    epoch_accuracy = sum(batch_accuracies) * 1.0 / len(batch_accuracies)
    epoch_loss = sum(batch_losses) * 1.0 / len(batch_losses)

    model.eval() # For computing on validation
    val_accuracy = get_accuracy_from_data(model, X_valid_tensor, y_valid_tensor)
    model.train() # Resetting to train mode
    
    print ('Epoch [{}/{}], Loss: {:.5f}, Average Train Acc (train mode): {:.5f}, Val Acc (eval mode): {:.5f}'.format(
        epoch+1, num_epochs, epoch_loss, epoch_accuracy, val_accuracy))

    epoch_losses.append(epoch_loss)
    epoch_val_acc.append(val_accuracy)
    epoch_train_acc.append(epoch_accuracy)
    
    # Periodic decaying of lr
    if (epoch+1) % epochs_per_decay == 0:
        lr *= lr_decay
        print('\nDecaying lr, new  rate = {:E}'.format(lr))
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# It Works!!! SAVE! SAVE! SAVE! SAVE! (It's 8 am now)
torch.save(model, 'time_only_cnn.pt')

# Move Test to GPU for speed
X_test_tensor, y_test_tensor = torch.tensor(X_test, device=device, dtype=torch.float), \
    torch.tensor(y_test, device=device, dtype=torch.long)
test_accuracy = get_accuracy_from_data(model.eval(), X_test_tensor, y_test_tensor)
print('Final Testing Accuracy: {:.5f}'.format(test_accuracy))

# Set to True for Plots
plot_losses(epoch_losses, show=False, filename='time_only_cnn.png')
plot_accuracies(epoch_train_acc, epoch_val_acc, show=False, filename='time_only_acc.png')


