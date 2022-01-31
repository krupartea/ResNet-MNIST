import torch

from torchvision import datasets





# The built-in MNIST dataset doesn't have a validation subset,
# and also an albumentation transofrm can't be passed.
# Create the wrapping dataset to soleve this problem

class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self, stage='train', transform=None):

        self.stage = stage
        
        # validation subset will be derived from the training set
        if self.stage in ['train', 'val']:
            dataset = datasets.MNIST(root='', train=True, download=True)
        else:
            dataset = datasets.MNIST(root='', train=False, download=True)
        
        # train, validation and test splitting
        if self.stage == 'train':
            start = 0
            end = int(0.9 * len(dataset.data))
        elif self.stage == 'val':
            start = int(0.9 * len(dataset.data))
            end = len(dataset.data)
        else:  # train
            start = 0
            end = len(dataset.data)
        
        self.data = dataset.data[start:end]
        self.targets = dataset.targets[start:end]
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        data = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            data = self.transform(image=data.cpu().numpy())['image']
        
        return data, target