# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLineUNet(nn.Module):
    """
    BaseLineUNet class is the implementation of the U-Net model with Conv2D layers.
    Applies a series of Conv2D layers to the input tensor and returns the output tensor. 
    """
    def __init__(self, n_channels, first_downsample, second_downsample, third_downsample, fourth_downsample, \
                 fifth_downsample, sixth_downsample, first_upsample, second_upsample, third_upsample, fourth_upsample, fifth_upsample):
        """
        Parameters  :   n_channels (int) - Number of channels in the input tensor,
                        The rest of the parameters are the number of channels in the respective convolutional layers.

        Returns     :   None
        """
        
        super().__init__()

        self.n_channels = n_channels
        self.first_downsample = first_downsample
        self.second_downsample = second_downsample
        self.third_downsample = third_downsample
        self.fourth_downsample = fourth_downsample
        self.fifth_downsample = fifth_downsample
        self.sixth_downsample = sixth_downsample
        self.first_upsample = first_upsample
        self.second_upsample = second_upsample
        self.third_upsample = third_upsample
        self.fourth_upsample = fourth_upsample
        self.fifth_upsample = fifth_upsample

        # Conv2D layers
        self.first_block = DoubleConv(self.n_channels, self.first_downsample)
        self.down1 = DownSample(self.first_downsample, self.second_downsample)
        self.down2 = DownSample(self.second_downsample, self.third_downsample)
        self.down3 = DownSample(self.third_downsample, self.fourth_downsample)
        self.down4 = DownSample(self.fourth_downsample, self.fifth_downsample)
        self.down5 = DownSample(self.fifth_downsample, self.sixth_downsample)
        self.up1 = UpSample(self.sixth_downsample + self.fifth_downsample, self.first_upsample)
        self.up2 = UpSample(self.first_upsample + self.fourth_downsample, self.second_upsample)
        self.up3 = UpSample(self.second_upsample + self.third_downsample, self.third_upsample)
        self.up4 = UpSample(self.third_upsample + self.second_downsample, self.fourth_upsample)
        self.up5 = UpSample(self.fourth_upsample + self.first_downsample, self.fifth_upsample)
        self.last_block = DoubleConv(self.fifth_upsample, self.n_channels)

    def forward(self, input_tensor):
        """
        Forward call for the BaseLineUNet class - Applies a series of Conv2D layers to the input tensor and returns the output tensor.
        Parameters  :   input_tensor (torch.Tensor) - Input tensor to the model
        Returns     :   output_tensor (torch.Tensor) - Output tensor from the model
        """
        x1 = self.first_block(input_tensor)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.up1(x6, x5)
        x8 = self.up2(x7, x4)
        x9 = self.up3(x8, x3)
        x10 = self.up4(x9, x2)
        x11 = self.up5(x10, x1)
        output_tensor = self.last_block(x11)
        return output_tensor

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
                )
        
    def forward(self, x):
        """
        Applies two Conv2D layers to the input tensor and returns the output tensor.
        Parameters  :   x (torch.Tensor) - Input tensor to the model
        Returns     :   x (torch.Tensor) - Output of the double convolutional layer
        """
        return self.double_conv(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
                )
        
    def forward(self, x):
        """
        Applies a MaxPool2D layer followed by a double convolutional layer to the input tensor and returns the output tensor.
        Parameters  :   x (torch.Tensor) - Input tensor to the model
        Returns     :   x (torch.Tensor) - Output of the downsample layer
        """
        return self.maxpool_conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = DoubleConv(in_channels, out_channels, in_channels//2)
        
    def forward(self, x1, x2):
        """
        Applies an Upsample layer followed by a double convolutional layer to the input tensor and returns the output tensor.
        Parameters  :   x1 (torch.Tensor) - Input tensor 
                        x2 (torch.Tensor) - Input tensor
        Returns     :   x (torch.Tensor) - Output of the upsample layer
        """
        x1 = self.upsample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class Handle(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
    
    def forward(self, input_tensor):
        """
        Applies the input tensor to the model and returns the output tensor - Handles the device placement of the model and the input tensor.
        Parameters  :   input_tensor (torch.Tensor) - Input tensor to the model
        Returns     :   output_tensor (torch.Tensor) - Output tensor from the model
        """
        if self.device == "cuda":
            self.model = self.model.cuda()
            input_tensor = input_tensor.cuda()
        
        if input_tensor.dim() < 4 or input_tensor.dim() < 3 or input_tensor.shape[2] < 32 or input_tensor.shape[3] < 32:
            raise ValueError("Please keep the following in mind while initializing the model:\n\
                             * Input tensor must be 4D or 3D and must have at least one channel and one time step!\n \
                             * Input tensor's height and width must be at least 32!\n")
        
        output_tensor = self.model(input_tensor)
        return output_tensor