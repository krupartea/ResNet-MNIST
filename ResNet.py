import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        x += identity
        x = self.relu(x)
        
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # create ResNet layers
        # NOTE: name 'layers' can be confusing as it has three meanings in this code:
        # 1) layers as a number of convolutional layers for what the number at the end
        #    of a model name stands. For example, ResNet50 has 50 convolutional layers in it,
        #    including both initial (out of Block class) convolutional layers
        #    and convolutional layers which blocks are comprised of
        # 2) layers as a high-level ResNet parts,
        #    containing some number of blocks (there is four such parts).
        #    They differ with the number of blocks, the number of out channels and the stride.
        #    Such part fragmentation gives ability to reduce amount of code
        #    because blocks with the same parameters are generated in the loop
        # 3) layers as a ResNet class parameter,
        #    which is a list of four integers, defining the number of blocks
        #    in each ResNet layer (last word 'layer' is referred to the second meaning)
        
    
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.fc(x)
        
        return x
    
    
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        indentity_downsample = None
        layers  = []
        
        if stride != 1 or self.in_channels != out_channels * 4:
            indentity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1,
                                                           stride=stride),
                                                 nn.BatchNorm2d(out_channels*4))
        
        # append the first block manually (out of the loop)
        # to render its output to have same dimentions (C, W, H) as further blocks input
        # and no need to call identity downsample in further blocks
        layers.append(block(self.in_channels, out_channels, indentity_downsample, stride))
        self.in_channels = out_channels * 4
        
        for i in range(num_residual_blocks - 1):  # -1 because the first block is already appended
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)


def ResNetMNIST50(img_channels=1, num_classes=10):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)