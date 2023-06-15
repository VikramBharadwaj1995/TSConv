import torch
from UNet import UNet
import yaml
import pytest

# Reading input params from config file
with open("parameters.yaml", "r") as f:
    params = yaml.safe_load(f)
    n_channels = int(params["init_params"]["n_channels"])
    first_downsample = int(params["init_params"]["first_downsample"])
    second_downsample = int(params["init_params"]["second_downsample"])
    third_downsample = int(params["init_params"]["third_downsample"])
    fourth_downsample = int(params["init_params"]["fourth_downsample"])
    fifth_downsample = int(params["init_params"]["fifth_downsample"])
    sixth_downsample = int(params["init_params"]["sixth_downsample"])
    first_upsample = int(params["init_params"]["first_upsample"])
    second_upsample = int(params["init_params"]["second_upsample"])
    third_upsample = int(params["init_params"]["third_upsample"])
    fourth_upsample = int(params["init_params"]["fourth_upsample"])
    fifth_upsample = int(params["init_params"]["fifth_upsample"])

def test_model_output_shape():
    x = torch.randn(1, 1, 256, 256)
    model = UNet(n_channels, first_downsample, second_downsample, third_downsample, fourth_downsample, \
                 fifth_downsample, sixth_downsample, first_upsample, second_upsample, third_upsample, fourth_upsample, fifth_upsample, use_tsconv=True, use_cuda=True)
    output = model(x)

    assert output.shape == (1, 1, 256, 256)

def test_UNet():
    input_tensor = torch.randn((1, 1, 256, 256))
    model = UNet(n_channels, first_downsample, second_downsample, third_downsample, fourth_downsample, \
                 fifth_downsample, sixth_downsample, first_upsample, second_upsample, third_upsample, fourth_upsample, fifth_upsample, use_tsconv=True, use_cuda=True)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)

if __name__ == "__main__":
    test_UNet()
    test_model_output_shape()