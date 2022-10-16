import torch
import torch.nn as nn
print(torch.__version__)

# Architecture, M means maxpool
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# Then flatten and 4096x4096x1000 Linear layers

class VGG_net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16)
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096), # 224(2**5) = 7 this is from 5 maxpooling layers
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1) # flatten
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers=[]
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int: # conv layer
                out_channels = x

                layers += [nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size = (3,3), stride = (1,1), padding = (1,1)),
                nn.BatchNorm2d(x), # not in original VGG paper because it was not invented at that time
                nn.ReLU()]

                in_channels = x

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
    
        return nn.Sequential(*layers)

if __name__ == '__main__':
    device = torch.device('mps')
    model = VGG_net(in_channels=3, num_classes=1000).to(device)
    x = torch.randn(1,3,224,224).to(device) # single image input
    print(model(x).shape)