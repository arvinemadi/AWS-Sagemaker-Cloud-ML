#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import argparse
import logging
import sys
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device, phase = 'test'):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            preds = model(data)
            total_loss += criterion(preds, label).item()
            preds = preds.argmax(dim = 1, keepdim = True)
            correct += preds.eq(label.view_as(preds)).sum().item()
        
    total_loss = total_loss / len(test_loader.dataset)
    total_acc = float(correct) / len(test_loader.dataset)
    
    
    logger.info(
        "\nTest set: Average loss: {:.4f}, Test Accuracy: {}%\n".format(
            total_loss, 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Starting the training...")
    model.train()
    index = 0
    for data_, label_ in train_loader:
        data_ = data_.to(device)
        label_ = label_.to(device)
        optimizer.zero_grad()
        preds = model(data_)
        loss = criterion(preds, label_)
        loss.backward()
        optimizer.step()
        index += 1
        if index % 100 == 0:
            logger.info(
                "Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    index * len(data_),
                    len(train_loader.dataset),
                    100.0 * index / len(train_loader),
                    loss.item(),
                )
            )
    
    
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                    nn.Linear(num_features, 256),
                    nn.ReLU(),
                    nn.Linear(256, 133) 
                    )
    
    return model

def create_data_loader(datapath, batch_size, type_ = 'train'):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    transform = transforms.Compose([])
    shuffle = False;
    if type_ == 'train':
        transform = transforms.Compose([
                                        transforms.RandomResizedCrop((224, 224)),                            
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                                       )
        shuffle = True
    else:
        transform = transforms.Compose([
                                       transforms.Resize((224, 224)),                            
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                                      )
    
    data = torchvision.datasets.ImageFolder(root = datapath, transform = transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = shuffle)
        
    
    return data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Available device is: {}".format(device))
    model = net()
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    
    loss_criterion = nn.CrossEntropyLoss(ignore_index = 133)
    optimizer = optim.Adam(model.fc.parameters(), lr = args.lr)
    
    train_loader = create_data_loader(args.datapath + '/train', args.batch_size)
    test_loader = create_data_loader(args.datapath + '/test', args.batch_size, type_ = 'test')
    val_loader = create_data_loader(args.datapath + '/valid', args.batch_size, type_ = 'test')
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, loss_criterion, optimizer, device)
        test(model, val_loader, loss_criterion, device)
    
    
      
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), args.modelpath + '/model.pth')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1.0, 
        metavar="LR", 
        help="learning rate (default: 1.0)"
    )
    
    parser.add_argument(
        '--datapath', 
        type=str, 
        default=os.environ["SM_CHANNEL_TRAIN"]
    )
    
    parser.add_argument(
        '--modelpath', 
        type=str, 
        default=os.environ['SM_MODEL_DIR']
    )
    
    parser.add_argument(
        '--outpath', 
        type=str, 
        default=os.environ['SM_OUTPUT_DATA_DIR']
    )
    
    
    
    args=parser.parse_args()
    print(args)
    
    main(args)
