import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse


# TODO: Define your transforms for the training, validation, and testing sets
def load_data(train_dir, valid_dir, test_dir):
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    testing_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    training_data = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_data = datasets.ImageFolder(test_dir, transform=testing_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainingloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
    testingloader = torch.utils.data.DataLoader(testing_data, batch_size=64, shuffle=True)

    class_mapping = training_data.class_to_idx

    return class_mapping, trainingloader, validationloader, testingloader


# Select device name
def get_device(gpu):
    device_name = "cpu"
    if gpu == True:
        if torch.mps.device_count() > 0:
            device_name = "mps"
        elif torch.cuda.is_available():
            device_name = "cuda"
    
    device = torch.device(device_name)
    return device


def generate_model(device, arch, dropout):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(OrderedDict([
                              ('fc1',nn.Linear(25088, 1024)),
                              ('relu1',nn.ReLU()),
                              ('dropout1',nn.Dropout(dropout)),
                              ('fc2',nn.Linear(1024, 512)),
                              ('relu2',nn.ReLU()),
                              ('dropout2',nn.Dropout(dropout)),
                              ('fc3',nn.Linear(512, 102)),
                              ('output',nn.LogSoftmax(dim=1))
                              ]))
    elif arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(OrderedDict([
                              ('fc1',nn.Linear(1024, 512)),
                              ('relu1',nn.ReLU()),
                              ('dropout1',nn.Dropout(dropout)),
                              ('fc2',nn.Linear(512, 256)),
                              ('relu2',nn.ReLU()),
                              ('dropout2',nn.Dropout(dropout)),
                              ('fc3',nn.Linear(256, 102)),
                              ('output',nn.LogSoftmax(dim=1))
                              ]))
    
    return model

def train_model(epochs, print_every, device, model, criterion, optimizer, trainingloader, validationloader):
    steps = 0
    training_loss = 0

    model.to(device)
    
    for epoch in range(epochs):
        for inputs, labels in trainingloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
    
            training_loss += loss.item()
            
            if steps % print_every == 0:
                accuracy = 0
                validation_loss = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        valid_loss = criterion(logps, labels)
                        
                        validation_loss += valid_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Training loss: {training_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validationloader):.3f}")
                training_loss = 0
                model.train()
    
    return model, optimizer

# TODO: Do validation on the test set
def test_accuracy(testingloader, device, model):
    accuracy = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in testingloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test Accuracy: {100 * (accuracy/len(testingloader)):.3f} % ")   


# TODO: Save the checkpoint 
def save_checkpoint(arch, epochs, dropout, model, class_mapping, optimizer, checkpoint_file_path):
    model.class_to_idx = class_mapping

    checkpoint = {'arch' : arch,
                  'epochs': epochs,
                  'dropout': dropout,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }

    if ".pth" not in checkpoint_file_path:
        checkpoint_file_path = checkpoint_file_path + ".pth"

    checkpoint_file_path = "./" + arch + "-" + checkpoint_file_path
    torch.save(checkpoint, checkpoint_file_path)
    return checkpoint_file_path

if __name__ == "__main__":
    # Gets Argument from CLI
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input-dir', default="./flowers/", type=str, help='input file path')
    parser.add_argument('--checkpoint', default='checkpoint.pth', type=str, help='checkpoint file path for saving')
    parser.add_argument('--gpu', default=True, type=bool, help='Device CPU or GPU')
    parser.add_argument('--arch', default='vgg16', type=str, choices=["vgg16", "densenet121"], help='chooses which pretrained model to use')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout value')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=7, type=int, help='Number of cycles given to train the model')
    
    args = parser.parse_args()
    
    data_dir = args.input_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    class_mapping, trainingloader, validationloader, testingloader = load_data(train_dir, valid_dir, test_dir)
    
    device = get_device(args.gpu)
    
    model = generate_model(device, args.arch, args.dropout)
    print(model)
    print("model generated")
    
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    model, optimizer = train_model(args.epochs, 20, device, model, criterion, optimizer, trainingloader, validationloader)
    print("training complete")

    test_accuracy(testingloader, device, model)
    print("testing complete")

    checkpoint_file_path = save_checkpoint(args.arch, args.epochs, args.dropout, model, class_mapping, optimizer, args.checkpoint)
    print("checkpoint saved", checkpoint_file_path)


