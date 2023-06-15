import torch
import torch.nn as nn
import BaseLineUNet, TSCUNet

class UNet(nn.Module):
    """
    UNet class - Initializes two UNet models - one with Conv2D layers and one with TSConv layers based on the input parameter use_tsconv.
    Usage of use_tsconv:
        - If use_tsconv is set to True, the model will use TSConv layers and will ignore the Conv2D layers - Total number of parameters: 4,547 (1/3 of the baseline model)
        - If use_tsconv is set to False, the model will use Conv2D layers and will ignore the TSConv layers - Total number of parameters: 13,337
    """
    def __init__(self, n_channels, first_downsample, second_downsample, third_downsample, fourth_downsample, \
                 fifth_downsample, sixth_downsample, first_upsample, second_upsample, third_upsample, fourth_upsample, fifth_upsample, use_tsconv=False, use_cuda=False):
        super().__init__()
        
        self.use_tsconv = use_tsconv
        self.tsconv_unet = None
        self.base_line_unet = None
        self.tsconv_handle = None
        self.baseline_handle = None
        self.use_cuda = use_cuda

        if use_tsconv:
            self.tsconv_unet = TSCUNet.TSCUNet(n_channels, first_downsample, second_downsample, third_downsample, fourth_downsample, \
                    fifth_downsample, sixth_downsample, first_upsample, second_upsample, third_upsample, fourth_upsample, fifth_upsample)
            self.tsconv_handle = TSCUNet.Handle(self.tsconv_unet, self.use_cuda)
        else:
            self.base_line_unet = BaseLineUNet.BaseLineUNet(n_channels, first_downsample, second_downsample, third_downsample, fourth_downsample, \
                    fifth_downsample, sixth_downsample, first_upsample, second_upsample, third_upsample, fourth_upsample, fifth_upsample)
            self.baseline_handle = BaseLineUNet.Handle(self.base_line_unet, self.use_cuda)
        

    def forward(self, input_tensor):
        """
        Applies either the TSConv or Conv2D layers to the input tensor based on the use_tsconv parameter.
        Parameters  :   input_tensor (torch.Tensor) - Input tensor to the model
        Returns     :   output_tensor (torch.Tensor) - Output tensor from the model
        """

        if self.use_tsconv:
            # Use TSConv
            if self.tsconv_unet is None:
                raise ValueError("TSConv U-Net is not initialized! - Please set use_tsconv=True")
            else:
                output_tensor = self.tsconv_handle(input_tensor)
        else:
            # Use Conv2D
            if self.base_line_unet is None:
                raise ValueError("Conv2D U-Net is not initialized! - Please set use_tsconv=False")
            else:
                output_tensor = self.baseline_handle(input_tensor)
        
        return output_tensor