import torch
import torch.nn as nn
import torch.nn.functional as F

class TSCUNet(nn.Module):
    """
    TSCUNet class is the implementation of the U-Net model with TSConv2D layers.
    Applies a series of TSConv2D layers to the input tensor and returns the output tensor.
    """
    def __init__(self, n_channels, first_downsample, second_downsample, third_downsample, fourth_downsample, \
                 fifth_downsample, sixth_downsample, first_upsample, second_upsample, third_upsample, fourth_upsample, fifth_upsample):
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

        # TSConv2D layers
        self.first_block = DoubleTSConv(self.n_channels, self.first_downsample)
        self.down1 = DownSampleTSC(self.first_downsample, self.second_downsample)
        self.down2 = DownSampleTSC(self.second_downsample, self.third_downsample)
        self.down3 = DownSampleTSC(self.third_downsample, self.fourth_downsample)
        self.down4 = DownSampleTSC(self.fourth_downsample, self.fifth_downsample)
        self.down5 = DownSampleTSC(self.fifth_downsample, self.sixth_downsample)
        self.up1 = UpSampleTSC(self.sixth_downsample + self.fifth_downsample, self.first_upsample)
        self.up2 = UpSampleTSC(self.first_upsample + self.fourth_downsample, self.second_upsample)
        self.up3 = UpSampleTSC(self.second_upsample + self.third_downsample, self.third_upsample)
        self.up4 = UpSampleTSC(self.third_upsample + self.second_downsample, self.fourth_upsample)
        self.up5 = UpSampleTSC(self.fourth_upsample + self.first_downsample, self.fifth_upsample)
        self.last_block = DoubleTSConv(self.fifth_upsample, self.n_channels)

    def forward(self, input_tensor):
        """
        Forward call for the TSCUNet class - Applies a series of TSConv2D layers to the input tensor and returns the output tensor.
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
    

class DoubleTSConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
                TSConv(in_channels, mid_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                nn.ReLU(inplace=True),
                TSConv(mid_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                nn.ReLU(inplace=True)
                )
        
    def forward(self, x):
            """
            Forward call for the DoubleTSConv class - Applies a series of TSConv2D layers to the input tensor and returns the output tensor.
            Parameters  :   x (torch.Tensor) - Input tensor to the model
            Returns     :   output_tensor (torch.Tensor) - Output tensor from the DoubleTSConv class after applying the TSConv2D layers
            """
            return self.double_conv(x)
        
class TSConv(nn.Module):
    """
    TSConv class is the implementation of the Time-Shifted Convolutional Layer.
    Applies a series of Time-Shifted Convolutional Layers to the time shifted input tensor and returns the output tensor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.static_channels = in_channels // 2
        self.dynamic_channels = in_channels - self.static_channels
        self.forward_dynamic = self.dynamic_channels // 2
        self.backward_dynamic = self.dynamic_channels - self.forward_dynamic
        
        self.forward_shift = ForwardShift(self.forward_dynamic)
        self.backward_shift = BackwardShift(self.backward_dynamic)

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, input_tensor):
        """
        The input tensor is split into static and dynamic channels. The dynamic channels are further split into forward and backward dynamic channels. Each 
        dynamic channel is then time shifted and concatenated with the static channels. The concatenated tensor is then passed through a Conv2D layer.

        Parameters  :   input_tensor (torch.Tensor) - Input tensor to the model
        Returns     :   output_tensor (torch.Tensor) - Output tensor from the TSConv class after applying the Time-Shifted Convolutional Layers
        """
        static = input_tensor[:, :self.static_channels, :, :]
        dynamic = input_tensor[:, self.static_channels:, :, :]
        forward = dynamic[:, :self.forward_dynamic, :, :]
        backward = dynamic[:, self.forward_dynamic:, :, :]
        
        forward = self.forward_shift(forward)
        backward = self.backward_shift(backward)
        
        dynamic = torch.cat((forward, backward), dim=1)
        input_tensor = torch.cat((static, dynamic), dim=1)
        
        return self.conv2d(input_tensor)

class DownSampleTSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleTSConv(in_channels, out_channels)
                )
        
    def forward(self, x):
        """
        Performs a MaxPool2D operation on the input tensor and applies a series of TSConv2D layers to the output tensor.
        Parameters  :   x (torch.Tensor) - Input tensor to the model
        Returns     :   output_tensor (torch.Tensor) - Output tensor from the DownSampleTSC class after applying the TSConv2D layers
        """
        return self.maxpool_conv(x)

class UpSampleTSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = DoubleTSConv(in_channels, out_channels, in_channels//2)
        
    def forward(self, x1, x2):
        """
        Performs an Upsample operation on the input tensor and applies a series of TSConv2D layers to the output tensor.
        Here, I have also used the padding technique to handle the mismatch in the height and width of the input and output tensors.
        I was facing issues while using the padding parameter in the nn.Upsample class, resolved it with refrence to this link: 
        https://github.com/milesial/Pytorch-UNet/blob/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet/unet_parts.py (Official implementation of the U-Net model)

        Parameters  :   x1 (torch.Tensor) - Input tensor to the model
                        x2 (torch.Tensor) - Input tensor to the model
        Returns     :   output_tensor (torch.Tensor) - Output tensor from the UpSampleTSC class after applying the TSConv2D layers
        """
        x1 = self.upsample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Applies a Conv2D layer to the input tensor and returns the single channel output tensor.
        Parameters  :   x (torch.Tensor) - Input tensor to the model
        Returns     :   output_tensor (torch.Tensor) - Output tensor from the OutConv class after applying the Conv2D layer
        """
        return self.conv(x)
    
class ForwardShift(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
    def forward(self, input_tensor):
        """
        The forward shift operation implemented as detailed in the PDF, with the following steps:
        1. Pad the input tensor with zeros on the left side
        2. Roll the tensor by 1 step to the right
        3. Remove the first column of the tensor

        Parameters  :   input_tensor (torch.Tensor) - Input tensor to the model
        Returns     :   output_tensor (torch.Tensor) - Output tensor from the ForwardShift class after applying the forward shift operation
        """
        input_tensor = F.pad(input_tensor, (1, 0), "constant", 0)
        output_tensor = torch.roll(input_tensor, 1, 3)
        return output_tensor[:, :, :, 1:]
    
class BackwardShift(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
    def forward(self, input_tensor):
        """
        The backward shift operation implemented as detailed in the PDF, with the following steps:
        1. Pad the input tensor with zeros on the right side
        2. Roll the tensor by 1 step to the left
        3. Remove the last column of the tensor
        
        Parameters  :   input_tensor (torch.Tensor) - Input tensor to the model
        Returns     :   output_tensor (torch.Tensor) - Output tensor from the BackwardShift class after applying the backward shift operation
        """
        input_tensor = F.pad(input_tensor, (0, 1), "constant", 0)
        output_tensor = torch.roll(input_tensor, -1, 3)
        return output_tensor[:, :, :, :-1]
    
class Handle(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
    
    def forward(self, input_tensor):
        """
        Applies the model to the input tensor and returns the output tensor - Handles the device placement of the model and the input tensor.

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
        
