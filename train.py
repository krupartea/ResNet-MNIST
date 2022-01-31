import torch
import albumentations as A
from albumentations import pytorch
from dataset import MNISTDataset
from torch.utils.data import DataLoader
import argparse
from ResNet import ResNetMNIST50
import torch.nn as nn


def main(args):
    
    # select device
    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'
    
    # set augmentations
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=.02, scale_limit=.01, rotate_limit=10, border_mode=0, p=1),
        A.Normalize(mean=.1307, std=.3081),
        pytorch.transforms.ToTensorV2()
    ])
    
    # create train and validation datasets objects
    train_dataset = MNISTDataset(stage='train', transform=transform)
    val_dataset = MNISTDataset(stage='val')
    
    # create the dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.val_batch_size)
    
    # create the model object
    model = ResNetMNIST50().to(device)
    
    # create training uitilities: a loss function (criterion), an optimizer and a scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    epoch_val_losses = []  # create list to save mean validation batch loss for every epoch
                           # to save the best model (with the least validation loss)
    
    # train the model with validation
    for epoch in range(args.num_epochs):
        
        # training stage
        model.train()
        
        train_batch_losses = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            
            # move mini-batches to proper device
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            
            loss = criterion(output, target)
            
            train_batch_losses.append(loss.item())
            
            # cast models gradients to zero
            model.zero_grad()  # if all models parameters are passed to the optimizer,
                               # it is the same as optimizer.zero_grad()
            
            # caclulate gradients
            loss.backward()
            
            # optimize parameters
            optimizer.step()
            
            # print log info
            if batch_idx % args.log_interval == 0:
                print(f'Epoch {epoch+1}/{args.num_epochs}\nLoss: {loss.item():4f}\n')
        
        # validation stage
        model.eval()  # de-activate Dropout layers, make normalisation layers use running statistics
        
        val_batch_losses = []
        
        for batch_idx, (data, target) in enumerate(train_loader):

            # move mini-batches to proper device
            data = data.to(device)
            target = target.to(device)
                        
            with torch.no_grad():  # disable the gradient computation
                output = model(data)
                loss = criterion(output, target)    
            val_batch_losses.append(loss.item())
        
        mean_val_loss = sum(val_batch_losses) / len(val_batch_losses)
        epoch_val_losses.append(mean_val_loss)
        
        print(f'Epoch {epoch+1}/{args.num_epochs}\nValidation loss: {mean_val_loss:4f}\n\n')
        
        # decrease the learning rate WRT validation loss
        scheduler.step(mean_val_loss)
        
        # save model
        if mean_val_loss == min(epoch_val_losses):
            torch.save(model.state_dict(), 'trained_models/least_loss.pt')
        torch.save(model.state_dict(), 'trained_models/latest.pt')

if __name__ == '__main__':
    
    # create the parser object
    parser = argparse.ArgumentParser()
    
    # create parser arguments
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--val-batch-size', type=int, default=1024)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--learning-rate', type=int, default=0.001)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--log-interval', type=int, default=20)
    
    # parse arguments from the command line
    args = parser.parse_args()
    
    main(args)