#This code is adopted from
#https://github.com/ivanvovk/WaveGrad
import torch

from model.base import BaseModule
from model.linear_modulation import FeatureWiseAffine
from model.interpolation import InterpolationBlock
from model.layers import Conv1dWithInitialization


class BasicModulationBlock(BaseModule):
    """
    Linear modulation part of UBlock, represented by sequence of the following layers:
        - Feature-wise Affine
        - LReLU
        - 3x1 Conv
    """

    def __init__(self, n_channels, dilation):
        super(BasicModulationBlock, self).__init__()
        self.featurewise_affine = FeatureWiseAffine()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        # self.leaky_relu = torch.nn.LeakyReLU(0.2, inplace=True)
        self.convolution = Conv1dWithInitialization(in_channels=n_channels,
                                                    out_channels=n_channels,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=dilation,
                                                    dilation=dilation)

    def forward_new(self, x, scale, shift):
        return self.convolution(
            self.leaky_relu(self.featurewise_affine(
                x,
                scale,
                shift,
            ), ), )

    def forward(self, x, scale, shift):
        outputs = self.featurewise_affine(x, scale, shift)
        assert not torch.any(
            torch.isnan(outputs)), "outputs1_featurewise_affine"
        # input()
        outputs = self.leaky_relu(outputs)
        if torch.any(torch.isnan(outputs)):
            input("outputs2_leaky_relu")
            # input()
        outputs = self.convolution(outputs)
        for name, weight in self.convolution.named_parameters():
            assert not torch.any(torch.isnan(
                weight)), f"upsampling_basicmodulationblock_forward{name}"
        if torch.any(torch.isnan(outputs)):
            input("outputs3_convolution")
            # input()

        return outputs


class UpsamplingBlock(BaseModule):

    def __init__(self, in_channels, out_channels, factor, dilations):
        super(UpsamplingBlock, self).__init__()
        self.first_block_main_branch = torch.nn.ModuleDict({
            'upsampling':
            torch.nn.Sequential(*[
                torch.nn.LeakyReLU(0.2),
                # torch.nn.LeakyReLU(0.2, inplace=True),
                InterpolationBlock(
                    scale_factor=factor, mode='linear', align_corners=False),
                Conv1dWithInitialization(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=dilations[0],
                                         dilation=dilations[0])
            ]),
            'modulation':
            BasicModulationBlock(out_channels, dilation=dilations[1])
        })
        self.first_block_residual_branch = torch.nn.Sequential(*[
            Conv1dWithInitialization(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=1,
                                     stride=1),
            InterpolationBlock(
                scale_factor=factor, mode='linear', align_corners=False)
        ])
        self.second_block_main_branch = torch.nn.ModuleDict({
            f'modulation_{idx}':
            BasicModulationBlock(out_channels, dilation=dilations[2 + idx])
            for idx in range(2)
        })

    def forward_new(self, x, scale, shift):
        # First upsampling residual block
        outputs = self.first_block_main_branch['modulation'](
            self.first_block_main_branch['upsampling'](x),
            scale,
            shift,
        ) + self.first_block_residual_branch(x)

        # Second residual block
        outputs += self.second_block_main_branch['modulation_1'](
            self.second_block_main_branch['modulation_0'](
                outputs,
                scale,
                shift,
            ), scale, shift)
        return outputs

    def forward(self, x, scale, shift):
        # First upsampling residual block
        outputs = self.first_block_main_branch['upsampling'](x)
        if torch.any(torch.isnan(x)):
            print("x", torch.isnan(x.sum(axis=0)))
            # input()
        if torch.any(torch.isnan(shift)):
            print("shift", torch.isnan(shift.sum(axis=0)))
            # input()
        if torch.any(torch.isnan(scale)):
            print("scale", torch.isnan(scale.sum(axis=0)))
            # input()
        if torch.any(torch.isnan(outputs)):
            print("outputs0", torch.isnan(outputs.sum(axis=0)))
            # input()
        outputs = self.first_block_main_branch['modulation'](outputs, scale,
                                                             shift)
        if torch.any(torch.isnan(outputs)):
            print("outputs1", torch.isnan(outputs.sum(axis=0)))
            # input()
        outputs = outputs + self.first_block_residual_branch(x)
        if torch.any(torch.isnan(outputs)):
            print("outputs2", torch.isnan(outputs.sum(axis=0)))
            # input()

        # Second residual block
        residual = self.second_block_main_branch['modulation_0'](outputs,
                                                                 scale, shift)
        if torch.any(torch.isnan(residual)):
            print("residual", torch.isnan(residual.sum(axis=0)))
            # input()
        outputs = outputs + self.second_block_main_branch['modulation_1'](
            residual, scale, shift)
        if torch.any(torch.isnan(outputs)):
            print("outputs3", torch.isnan(outputs.sum(axis=0)))
            # input()
        return outputs


class UpsamplingLargeBlock(BaseModule):

    def __init__(self, in_channels, out_channels, factor, dilations):
        super(UpsamplingLargeBlock, self).__init__()
        self.upsampling_basic_block = UpsamplingBlock(in_channels,
                                                      out_channels, factor,
                                                      dilations)
        self.first_block_main_branch = torch.nn.ModuleDict({
            'noupsampling':
            torch.nn.Sequential(*[
                torch.nn.LeakyReLU(0.2),
                # torch.nn.LeakyReLU(0.2, inplace=True),
                Conv1dWithInitialization(in_channels=out_channels,
                                         out_channels=out_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=dilations[0],
                                         dilation=dilations[0]),
            ]),
            'modulation':
            BasicModulationBlock(out_channels, dilation=dilations[1])
        })
        self.first_block_residual_branch = torch.nn.Sequential(*[
            Conv1dWithInitialization(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
            ),
        ])
        self.second_block_main_branch = torch.nn.ModuleDict({
            f'modulation_{idx}':
            BasicModulationBlock(out_channels, dilation=dilations[2 + idx])
            for idx in range(2)
        })

    def forward_new(self, x, scale, shift):
        # Frist Upsampling block
        x = self.upsampling_basic_block(x, scale, shift)
        # Second Block with no upsampling
        # Second-First residual block
        outputs = self.first_block_main_branch['modulation'](
            self.first_block_main_branch['noupsampling'](x),
            scale,
            shift,
        ) + self.first_block_residual_branch(x)

        # Second-Second residual block
        outputs += self.second_block_main_branch['modulation_1'](
            self.second_block_main_branch['modulation_0'](
                outputs,
                scale,
                shift,
            ),
            scale,
            shift,
        )
        return outputs

    def forward(self, x, scale, shift):
        # Frist Upsampling block
        x = self.upsampling_basic_block(x, scale, shift)
        # Second Block with no upsampling
        # Second-First residual block
        outputs = self.first_block_main_branch['noupsampling'](x)
        outputs = self.first_block_main_branch['modulation'](outputs, scale,
                                                             shift)
        outputs = outputs + self.first_block_residual_branch(x)

        # Second-Second residual block
        residual = self.second_block_main_branch['modulation_0'](outputs,
                                                                 scale, shift)
        outputs = outputs + self.second_block_main_branch['modulation_1'](
            residual, scale, shift)
        return outputs
