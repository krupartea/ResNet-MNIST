import torch
import argparse
from dataset import MNISTDataset
from torch.utils.data import DataLoader
from ResNet import ResNetMNIST50
from ignite.metrics import Accuracy
import albumentations as A
from albumentations import pytorch


def main(args):
    
    # define preprocessing
    transform = A.Compose([
        A.Normalize(mean=.1307, std=.3081),
        pytorch.transforms.ToTensorV2()
    ])
    
    # select device
    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'
    
    # create test dataset
    test_dataset = MNISTDataset(stage='test', transform=transform)
    
    # create test dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size)

    # create the model instance
    model = ResNetMNIST50().to(device)
    
    # load model weights
    model.load_state_dict(torch.load(args.model_path))
    
    # move model to proper device
    model.to(device)
    
    # model to the evaluation mode
    model.eval()
    
    # create the accuracy metric instance
    accuracy = Accuracy()
    
    # reset its data
    accuracy.reset()

    for data, target in test_loader:

            # move mini-batches to proper device
            data = data.to(device)
            target = target.to(device)
            
            # calculate the output
            with torch.no_grad(): # disable the gradient computation
                output = model(data)
            
            accuracy.update((output, target))
            
    print(f'Test accuracy: {accuracy.compute():.4f}')


if __name__ == '__main__':
    
    # create the parser instance
    parser = argparse.ArgumentParser()
    
    # create parser arguments
    parser.add_argument('--test-batch-size', type=int, default=1024)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--model-path')
    
    # parse arguments from the command line
    args = parser.parse_args()
    
    main(args)