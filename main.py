from random import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import time, os
import AlexNet


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main methods for the train dataloader, also optionally can return validation loader
def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True, num_workers=1, rp_b=False,rp=False):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # RP does not work for augment currently (ignores augment as we do not use in our experiments)
    # RP_B flag means that we use bfloat16 format
    if rp_b:
        train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
            transforms.ConvertImageDtype(torch.bfloat16),
        ])
        valid_transform = transforms.Compose([
              transforms.Resize((227,227)),
              transforms.ToTensor(),
              normalize,
              transforms.ConvertImageDtype(torch.bfloat16),
        ])
    elif rp:
        train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
            transforms.ConvertImageDtype(torch.float16),
        ])
        valid_transform = transforms.Compose([
              transforms.Resize((227,227)),
              transforms.ToTensor(),
              normalize,
              transforms.ConvertImageDtype(torch.float16),
        ])
    else: # Normal full precision, no half precision needed and hence data loader does not apply this transform
        valid_transform = transforms.Compose([
              transforms.Resize((227,227)),
              transforms.ToTensor(),
              normalize,
        ])
        if augment:
            train_transform = transforms.Compose([
              transforms.RandomCrop(32, padding=4),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              normalize,
            ])
        else:
            train_transform = transforms.Compose([
              transforms.Resize((227,227)),
              transforms.ToTensor(),
              normalize,
            ])


    # load the dataset
    #train_dataset = datasets.CIFAR10(
   #     root=data_dir, train=True,
   #     download=True, transform=train_transform,
   # )

    #valid_dataset = datasets.CIFAR10(
    #    root=data_dir, train=True,
    #    download=True, transform=valid_transform,
    #)

    # Use the imagenette2-160  data instead that was downloaded in previous steps
    train_dataset = datasets.ImageFolder(root='imagenette2-160/train/', transform=train_transform)
    valid_dataset = datasets.ImageFolder(root='imagenette2-160/val/', transform=valid_transform)

    # find the indices for train and validation


    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    return (train_loader, valid_loader)

# Method that returns the test data loader, so that we can evaluate accuracy
def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,rp=False,rp_b=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    if rp_b:
      transform = transforms.Compose([
          transforms.Resize((227,227)),
          transforms.ToTensor(),
          normalize,
          transforms.ConvertImageDtype(torch.bfloat16),
        ])
    elif rp:
      transform = transforms.Compose([
              transforms.Resize((227,227)),
              transforms.ToTensor(),
              normalize,
              transforms.ConvertImageDtype(torch.float16),
      ])
    else:
      transform = transforms.Compose([
          transforms.Resize((227,227)),
          transforms.ToTensor(),
          normalize,
      ])

    #dataset = datasets.CIFAR10(
    #    root=data_dir, train=False,
    #    download=True, transform=transform,
    #)

    # Use the validation data from ImageNette2
    dataset = datasets.ImageFolder(root='imagenette2-160/val/', transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


# CIFAR10 dataset 
#train_loader, valid_loader = get_train_valid_loader(data_dir = './data', batch_size = 64, augment = False, random_seed = 1)

#test_loader = get_test_loader(data_dir = './data',
#                              batch_size = 64)



#
# Wandb setup
#

!pip install wandb -Uq

import wandb, pprint
wandb.login()

sweep_config = {
    'method': 'random',
  }

metric = {
    'name': 'loss',
    'goal': 'minimize'
  }

sweep_config['metric'] = metric

parameters_dict = {
    'num_workers': {
        'values': [1, 2, 4, 8]#, 16]
        },
    'rp_b': {# bfloat16 for many previous sweeps, this is 'rp' for float16
        'values': [True, False]
        },
    'channels_last': {
        'values': [True, False]
        },
    'batch_size': {
        'values': [64, 128, 256, 512]
        },
    'learning_rate': {
        'values': [0.005, 0.001, 0.0005]
        },
}

sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)

# Get new sweep ID
sweep_id = wandb.sweep(sweep_config, project="hpc-proj") #"<wandb_entity?>/hpc-proj/swt2n8jz"


#
# Training
#
#
# Let us study the model performance below for our experiments
#
# Function to be called from wandb for setting up config parameters and running the training of modules
def train_wandb(config=None):
  #
  # Get wandb config
  #
  with wandb.init(config=config):
    config = wandb.config

    # Store each epoch's timings in arrays so that in the end we can average and return values 
    # (ignoring the first epoch for warmup reasons)
    dataLoaderTimeArr = []
    trainingTimeArr = []
    epochDataLoaderTimeArr = []
    epochTrainingTimeArr = []
    epochTimeArr = []

    #RP - float.16, RP_B - bfloat.16

    model = AlexNet(num_classes).to(device)
    NUM_WORKERS = wandb.config.num_workers#4

    # NOTE: For RP variations, two different sweeps are important, as both cannot be true at the same time
    # Toggle the following few lines so only one is active anytime
    #is_RP = wandb.config.rp #False
    is_RP_B = wandb.config.rp_b #False
    #if is_RP:
    #  model = model.to(dtype=torch.float16)
      #criterion = nn.MultiMarginLoss()#(log_target=False)
    if is_RP_B:
      model = model.to(dtype=torch.bfloat16)
    #if is_RP and is_RP_B: #NOTE: Choose only one RP type bfloat.16 or float.16
    #  if random() < 0.5:
    #    is_RP = False
    #  else:
    #    is_RP_B = False
    train_loader, valid_loader = get_train_valid_loader(data_dir = './data', batch_size = wandb.config.batch_size, augment = False, random_seed = 1,
                                                        num_workers=NUM_WORKERS,rp_b=is_RP_B)#rp=is_RP)#
    test_loader = get_test_loader(data_dir = './data', batch_size = 64, rp_b = is_RP_B)#_Brp=is_RP)#

    # Set channels last format if needed
    is_CHANNELS_LAST = wandb.config.channels_last #True
    if is_CHANNELS_LAST:
      model = model.to(memory_format=torch.channels_last)
    # Loss and optimizer definitions
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiMarginLoss()#(log_target=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.learning_rate, weight_decay = 0.005, momentum = 0.9)

    final_loss = 0.0

    #
    # Start training for num_epochs
    for epoch in range(num_epochs):
        begin_e = time.monotonic_ns()
        begin_i = begin_e
        for i, (images, labels) in enumerate(train_loader):  
            finish_i = time.monotonic_ns()
            dataLoaderTimeArr.append((finish_i-begin_i)/1000000000.0)

            #if is_RP:
            #  images = images.to(dtype=torch.float16)
            #  labels = labels.to(dtype=torch.int16)
            if is_CHANNELS_LAST:
              images = images.to(memory_format=torch.channels_last)

            begin_t = time.monotonic_ns()
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            #if is_RP:
            #  labels = labels.to(dtype=torch.long)
            # TODO some optimization possible with a criterion that accepts  half precision int labels

            # Calculate the loss
            loss = criterion(outputs, labels)
            # Just copy each time
            final_loss = loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            finish_t = time.monotonic_ns()
            trainingTimeArr.append((finish_t-begin_t)/1000000000.0)
            begin_i = time.monotonic_ns()
        finish_e = time.monotonic_ns()

        #print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        #              .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        epochTimeArr.append((finish_e-begin_e)/1000000000.0)
        epochDataLoaderTimeArr.append(np.sum(dataLoaderTimeArr))
        epochTrainingTimeArr.append(np.sum(trainingTimeArr))
        dataLoaderTimeArr = []
        trainingTimeArr = []

    wandb_run_id = wandb.run.id

    # Save model's final checkpoint
    directory = os.path.join("alexnet", 'wandb_{}'.format(wandb_run_id))
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({
        'iteration': num_epochs,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(directory, 'wandb_{}_{}_{}.tar'.format(wandb_run_id, epoch, 'checkpoint')))

    # Log in wandb the loss for this training instance run
    wandb.log({"loss": final_loss})


    # Log in wandb 
    wandb.log({"dataloader_time": np.mean(epochDataLoaderTimeArr[1:])})
    wandb.log({"training_time": np.mean(epochTrainingTimeArr[1:])})
    wandb.log({"epoch_time": np.mean(epochTimeArr[1:])})
    #wandb.log({"final_accuracy": final_accuracy})
    # Print time taken in epoch 2
    #if epochTimeArr:
        #print("Epoch timings: ", epochTimeArr[1])
        #print("DataLoader time: ", epochDataLoaderTimeArr[1])
        #print("Training time: ", epochTrainingTimeArr[1])
        #print("Epoch training loss: ", epoch_train_loss)
        #print("Epoch top-1 training accuracy: ", epoch_accuracy)

    # Perform testing accuracy evaluation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            #if is_RP:
            #  images = images.to(dtype=torch.float16)
            #  labels = labels.to(dtype=torch.int16)
            if is_CHANNELS_LAST:
              images = images.to(memory_format=torch.channels_last)

            outputs = model(images)
            #if is_RP:
            #  labels = labels.to(dtype=torch.long)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        wandb.log({"test_accuracy": 100*correct/total})

        #print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))   


wandb.agent(sweep_id, train_wandb, count=12)
