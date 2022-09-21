import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import json
import argparse
import logging
import sys
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import smdebug.pytorch as smd
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook
import time


    
def net():
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

def create_train_data_loader(datapath, batch_size):
    transform = transforms.Compose([
                                        transforms.RandomResizedCrop((224, 224)),                            
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                                       )
    
    data = torchvision.datasets.ImageFolder(root = datapath, transform = transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True)
        
    
    return data_loader

def create_test_data_loader(datapath, batch_size):
    transform = transforms.Compose([
                                       transforms.Resize((224, 224)),                            
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                                      )
    
    data = torchvision.datasets.ImageFolder(root = datapath, transform = transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = False)
        
    
    return data_loader


def test(model, test_loader, hook, device, criterion):
    print("Testing...")
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    total_loss = 0
    correct = 0
    loss_criterion = criterion
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            preds = model(data)
            total_loss += loss_criterion(preds, label).item()
            preds = preds.argmax(dim = 1, keepdim = True)
            correct += preds.eq(label.view_as(preds)).sum().item()
        
    total_loss = total_loss / len(test_loader.dataset)
    total_acc = float(correct) / len(test_loader.dataset)
    
    
    logger.info(
        "\nTest set: Average loss: {:.4f}, Test Accuracy: {}%\n".format(
            total_loss, 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(args, model, hook, device, criterion, optimizer):
    
    print("Hyperparameters: epoch: {}, lr: {}, batch size: {}".format(args.epochs, args.lr, args.batch_size))
    
    train_loader = create_train_data_loader(args.train_dir, args.batch_size)
    test_loader = create_test_data_loader(args.test_dir, args.batch_size)
    
    loss_criterion = criterion
    
    hook.register_loss(loss_criterion)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        hook.set_mode(modes.TRAIN)
        
        for index, (data_, label_) in enumerate(train_loader, 1):
            data_ = data_.to(device)
            label_ = label_.to(device)
            optimizer.zero_grad()
            preds = model(data_)
            loss = loss_criterion(preds, label_)
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                print(
                "Train epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    index * len(data_),
                    len(train_loader.dataset),
                    100.0 * index / len(train_loader),
                    loss.item(),
                    )
                )
        test(model, test_loader, hook, device, criterion)
    
    modelpath = os.path.join(args.model_dir, 'model.pth')
    logger.info("Saving the model to " + modelpath)
    torch.save(model.cpu().state_dict(), modelpath)


def model_fn(model_dir):
    logger.info('model_fn')
    device="cuda" if torch.cuda.is_available() else "cpu"
    model = net()
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model=nn.DataParallel(model)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Available device is: {}".format(device))
    model = net()
    model = model.to(device)
    
    
    
    print("Created the hook and registered it to the model!")
    
    criterion = nn.CrossEntropyLoss(ignore_index = 133)
    optimizer = optim.Adam(model.fc.parameters(), lr = args.lr)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(criterion)
    
    train(args, model = model, hook = hook, device = device, criterion = criterion, optimizer = optimizer)
      
    print("Training Finished...")
    
    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
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
        default=os.environ["SM_CHANNEL_TRAINING"]
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
    
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    
    main(parser.parse_args())
