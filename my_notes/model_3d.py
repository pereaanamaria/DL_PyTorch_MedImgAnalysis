import torch


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())
        
    def forward(self, X):
        return self.step(X)


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = DoubleConv(1, 32)
        self.layer2 = DoubleConv(32, 64)
        self.layer3 = DoubleConv(64, 128)
        self.layer4 = DoubleConv(128, 256)

        self.layer5 = DoubleConv(256 + 128, 128)
        self.layer6 = DoubleConv(128 + 64, 64)
        self.layer7 = DoubleConv(64 + 32, 32)
        self.layer8 = torch.nn.Conv3d(32, 3, 1)

        self.maxpool = torch.nn.MaxPool3d(2)

    
    def forward(self, x):
        
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)
                
        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)
                
        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)
         
        x4 = self.layer4(x3m)
                
        x5 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x4)  # Upsample with a factor of 2
        x5 = torch.cat([x5, x3], dim=1)  # Skip-Connection
        x5 = self.layer5(x5)
               
        x6 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x5)        
        x6 = torch.cat([x6, x2], dim=1)  # Skip-Connection    
        x6 = self.layer6(x6)
             
        x7 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x6)
        x7 = torch.cat([x7, x1], dim=1)       
        x7 = self.layer7(x7)
                
        return self.layer8(x7)