import click
import os
import torch
import time
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from workspace_utils import active_session

# Setup CLI functionality with click

@click.command()

@click.option('--save_dir', type=str, default=os.getcwd(), help='Save directory for model checkpoint. Defaults to current working directory.')
@click.option('--arch', type=click.Choice(['vgg13', 'vgg16']), help='Select architecture. Options are "vgg13" or "vgg16."')
@click.option('--learning_rate', type=float, default=0.001, help='Set the learning rate, default recommended.')
@click.option('--hidden_units', type=click.IntRange(102,4096), default=256, help='Set the number of hidden units, value between 102-4096.')
@click.option('--epochs', type=click.IntRange(2,50), default=10, help='Set the number of epochs to train, between 2 and 50.')
@click.option('--gpu', is_flag=True, help='Set this flag to train on GPU, if available.')

@click.argument('data_dir')


def train(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    
    # First, define helper functions
    
    def save_checkpoint(model, criterion, optimizer, num_epochs, filepath, arch, hidden_units):

        model.class_to_idx = image_datasets['train_dataset'].class_to_idx

        checkpoint = {'criterion': criterion.state_dict(),
                      'num_epochs': num_epochs,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'class_to_idx': model.class_to_idx,
                      'arch': arch,
                      'hidden_units': hidden_units
                     }
        
        path_and_name = filepath + '/' + 'model_checkpoint.pth'
        
        torch.save(checkpoint, path_and_name)
    
    def test_model(model, loader, criterion):
        with active_session():

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model.to(device)

            model.eval()

            since = time.time()

            test_loss = 0
            test_acc = 0
            steps = 0        

            with torch.no_grad():
                for images, labels in testloader:
                    steps += 1
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    batch_loss = criterion(log_ps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    test_acc += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"""
                Step: {steps+1}..
                Test Loss: {(test_loss/len(loader)):.3f}...
                Test Accuracy: {(test_acc/len(loader)):.3f}
                Time Elapsed: {((time.time() - since)//60):.0f}m {((time.time() - since) % 60):.0f}s
            """)
            
    # Check for GPU, inform user if unavailable
    if gpu:
        if torch.cuda.is_available():
            pass
        else:
            click.secho("No GPU available, training on CPU", fg='red')
    
    click.secho(f"""
    Beginning training with your provided settings:
    {'-'*20}
    data_dir: {data_dir}
    save_dir: {save_dir}
    arch: {arch}
    learning_rate: {learning_rate}
    hidden_units: {hidden_units}
    epochs: {epochs}
    gpu: {gpu}""", fg='green')
    
    # Setup directories based on input
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Build transforms and dataloaders
    
    data_transforms = {'train_transform': transforms.Compose([transforms.RandomRotation(30),
                                                          transforms.RandomResizedCrop(224),
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                               [0.229, 0.224, 0.225])]),
                  'valid_test_transform': transforms.Compose([transforms.Resize(255),
                                                              transforms.CenterCrop(224),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                                   [0.229, 0.224, 0.225])])}

    image_datasets = {'train_dataset': datasets.ImageFolder(train_dir, transform=data_transforms['train_transform']),
                     'valid_dataset': datasets.ImageFolder(valid_dir, transform=data_transforms['valid_test_transform']),
                     'test_dataset': datasets.ImageFolder(test_dir, transform=data_transforms['valid_test_transform'])}

    
    trainloader = torch.utils.data.DataLoader(image_datasets['train_dataset'], batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(image_datasets['valid_dataset'], batch_size=64)
    testloader = torch.utils.data.DataLoader(image_datasets['test_dataset'], batch_size=64)
    
    # Select model based on user's input for the --arch option
    
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    # Freeze params
    
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    # Build new classifier for our use case, bolt on to end of network. Take in hidden_units option
    
    new_classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(4096, hidden_units)),
                                       ('relu', nn.ReLU()),
                                       ('dropout', nn.Dropout()),
                                       ('fc2', nn.Linear(hidden_units, 102)),
                                       ('output', nn.LogSoftmax(dim=1))]))

    model.classifier[6] = new_classifier
    
    # Establish loss function and optimizer. Set LR based on user input
    
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier[6].parameters(), lr=learning_rate)
    
    # Send model to GPU, if available
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Train the model using options from CLI input
    
    with active_session():

        since = time.time()

        steps = 0
        running_loss = 0
        updates = 5
        best_acc = 0

        train_losses, valid_losses = [], []

        for e in range(epochs):
            print(f"Epoch {e+1}/{epochs}")
            print('-'*20)
            for images, labels in trainloader:
                steps += 1
                acc_train = 0
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                log_ps = model(images)

                _, preds = torch.max(log_ps.data, 1)

                loss = criterion(log_ps, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                acc_train += torch.sum(preds == labels.data).item()/64

                if steps % updates == 0:
                    valid_loss = 0
                    valid_accuracy = 0
                    model.eval()

                    with torch.no_grad():
                        for images, labels in validloader:
                            images, labels = images.to(device), labels.to(device)
                            log_ps = model(images)
                            batch_loss = criterion(log_ps, labels)
                            valid_loss += batch_loss.item()

                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    train_losses.append(running_loss/len(trainloader))
                    valid_losses.append(valid_loss/len(validloader))

                    train_loss_out = running_loss/updates
                    train_accuracy_out = acc_train
                    valid_loss_out = valid_loss/len(validloader)
                    valid_accuracy_out = valid_accuracy/len(validloader)

                    print(f"""
                        Step: {steps+1}..
                        Training Loss: {train_loss_out:.3f}...
                        Training Accuracy: {train_accuracy_out:.3f}
                        Validation Loss: {valid_loss_out:.3f}...
                        Validation Accuracy: {valid_accuracy_out:.3f}
                        Time Elapsed: {((time.time() - since)//60):.0f}m {((time.time() - since) % 60):.0f}s
                    """)
                    
                    # save model checkpoint if these are best results

                    if valid_accuracy_out > best_acc:
                        best_acc = valid_accuracy_out
                        save_checkpoint(model, criterion, optimizer, epochs, save_dir, arch, hidden_units)
                        print("Most accurate results so far, checkpoint saved.")

                    running_loss = 0
                    acc_train = 0
                    model.train()
    
    # With training complete, run model on test set and save checkpoint in provided directory
    test_model(model, testloader, criterion)    
    
    click.secho('Model saved, run complete!', fg='green')
    
if __name__ == '__main__':
    train()