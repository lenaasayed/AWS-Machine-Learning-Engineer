#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os

import argparse

from torch.optim import lr_scheduler
from torchvision import datasets, models
#TODO: Import dependencies for Debugging andd Profiling
import time
from smdebug import modes
from smdebug.profiler.utils import str2bool
# from smdebug.pytorch import get_hook

def test(model,criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    batch_size = 32
    transform_test = transforms.Compose(
        [
            transforms.ToTensor()        ]
    )

    testset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data, 'dogImages/test/'),transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True
    )    
    
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects / len(test_loader)
    
    print("total_loss ",total_loss)
    print("total_acc ",total_acc)

    
def train(model,criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    batch_size = 32
    epoch = 10
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )

    transform_valid = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()        ]
    )

    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data, 'dogImages/train/'),transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True
    )

    validset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data, 'dogImages/valid/'),transform=transform_valid
    )
    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=False
    )

    epoch_times = []

#     if hook:
#         hook.register_loss(loss_optim)
    # train the model

    for i in range(epoch):
        print("START TRAINING")
#         if hook:
#             hook.set_mode(modes.TRAIN)
        start = time.time()
        model.train()
        train_loss = 0
        for _, (inputs, targets) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_optim(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print("START VALIDATING")
#         if hook:
#             hook.set_mode(modes.EVAL)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(validloader):
                outputs = model(inputs)
                loss = loss_optim(outputs, targets)
                val_loss += loss.item()

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        print(
            "Epoch %d: train loss %.3f, val loss %.3f, in %.1f sec"
            % (i, train_loss, val_loss, epoch_time))
    return model


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model
#     pass

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
 
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)            
#         torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )

            
    parser.add_argument( "--epochs", type = int, default = 2, metavar="N", help="number of epochs to train (default: 2)" )

    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args=parser.parse_args()
  
    main(args)
