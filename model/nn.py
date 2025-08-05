#This code is adopted from
#https://github.com/ivanvovk/WaveGrad
import numpy as np

import torch

from model.base import BaseModule
from model.layers import Conv1dWithInitialization
from model.upsampling import UpsamplingBlock as UBlock
from model.downsampling import DownsamplingBlock as DBlock
from model.linear_modulation import FeatureWiseLinearModulation as FiLM
from torch.profiler import profile, record_function, ProfilerActivity


class WaveGradNN(BaseModule):
    """
    WaveGrad is a fully-convolutional mel-spectrogram conditional
    vocoder model for waveform generation introduced in
    "WaveGrad: Estimating Gradients for Waveform Generation" paper (link: https://arxiv.org/pdf/2009.00713.pdf).
    The concept is built on the prior work on score matching and diffusion probabilistic models.
    Current implementation follows described architecture in the paper.
    """

    def __init__(self, hparams):
        super(WaveGradNN, self).__init__()
        # Building upsampling branch (mels -> signal)
        self.hparams = hparams
        self.ublock_preconv = Conv1dWithInitialization(
            in_channels=hparams.wavegrad.encode_channel,
            out_channels=hparams.wavegrad.upsample.preconv_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        #"""
        upsampling_in_sizes = [hparams.wavegrad.upsample.preconv_channel
                               ] + hparams.wavegrad.upsample.out_channels[:-1]
        """
        upsampling_in_sizes = hparams.wavegrad.upsample.out_channels
        #"""
        self.ublocks = torch.nn.ModuleList([
            UBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations,
            ) for in_size, out_size, factor, dilations in zip(
                upsampling_in_sizes,
                hparams.wavegrad.upsample.out_channels,
                hparams.wavegrad.scale_factors,
                hparams.wavegrad.upsample.dilations,
            )
        ])
        self.ublock_postconv = Conv1dWithInitialization(
            in_channels=hparams.wavegrad.upsample.out_channels[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Building downsampling branch (starting from signal)
        self.dblock_preconv = Conv1dWithInitialization(
            in_channels=1,
            out_channels=hparams.wavegrad.downsample.preconv_channel,
            # kernel_size=5,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        #"""
        downsampling_in_sizes = [
            hparams.wavegrad.downsample.preconv_channel
        ] + hparams.wavegrad.downsample.out_channels[:-1]
        """
        downsampling_in_sizes = hparams.wavegrad.downsample.out_channels
        #"""
        self.dblocks = torch.nn.ModuleList([
            DBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations,
            ) for in_size, out_size, factor, dilations in zip(
                downsampling_in_sizes,
                hparams.wavegrad.downsample.out_channels,
                hparams.wavegrad.scale_factors[1:][::-1],
                hparams.wavegrad.downsample.dilations,
            )
        ])

        # Building FiLM connections (in order of downscaling stream)
        film_in_sizes = [hparams.wavegrad.downsample.preconv_channel] \
                + list(hparams.wavegrad.downsample.out_channels)
        film_out_sizes = hparams.wavegrad.upsample.out_channels[::-1]
        film_factors = [1] + hparams.wavegrad.scale_factors[1:][::-1]
        
        self.films = torch.nn.ModuleList([
            FiLM(
                in_channels=in_size,
                out_channels=out_size,
                input_dscaled_by=np.prod(
                    film_factors[:i + 1]
                ),  # for proper positional encodings initialization
                linear_scale=hparams.ddpm.pos_emb_scale,
            )
            for i, (in_size,
                    out_size) in enumerate(zip(film_in_sizes, film_out_sizes))
        ])
    ###no gradient forward
    # @torch.no_grad()
    def forward_i(self, yn, hidden_rep, noise_level):
        """
        Computes forward pass of neural network.
        :param mels (torch.Tensor): mel-spectrogram acoustic features of shape [B, n_mels, T//hop_length]
        :param yn (torch.Tensor): noised signal `y_n` of shape [B, T]
        :param noise_level (float): level of noise added by diffusion
        :return (torch.Tensor): epsilon noise
        """
        # Prepare inputs
        # hidden_rep = hidden_rep.unsqueeze(0)
        print('line 120 hidden_rep',hidden_rep.shape)
        assert len(hidden_rep.shape) == 3  # B, n_mels, T
        # print(f"nn.py: yn.shape: {yn.shape}")
        # print(f"nn.py: yn.shape: {len(yn.shape)}")
        yn = yn.unsqueeze(1)
        
        assert len(yn.shape) == 3  # B, 1, T
        # Downsampling stream + Linear Modulation statistics calculation
        statistics = []
        try:
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #     with record_function("Conv1D Operation"):
            dblock_outputs = self.dblock_preconv.forward(yn)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        except Exception as e:
            print(yn.shape)
            raise e
        # print("dblock_outputs", dblock_outputs.shape)
        # dblock_outputs torch.Size([1, 32, 103800]),noise_leveltorch.Size([1, 1])
        # dblock_outputs torch.Size([8, 32, 99300]),noise_leveltorch.Size([8, 8])
        # input(
        #     f"dblock_outputs {dblock_outputs.shape},noise_level {noise_level.shape}"
        # )
        scale, shift = self.films[0](x=dblock_outputs, noise_level=noise_level)
        statistics.append([scale, shift])
        """
        for dblock, film in zip(self.dblocks, self.films[1:]):
            dblock_outputs = dblock(dblock_outputs)
            scale, shift = film(x=dblock_outputs, noise_level=noise_level)
            statistics.append([scale, shift])
        """
        for i in range(len(self.dblocks)):
            dblock_outputs = self.dblocks[i].forward(dblock_outputs)
            scale, shift = self.films[i + 1].forward(
                x=dblock_outputs,
                noise_level=noise_level,
            )
            statistics.append([scale, shift])
        #"""
        statistics = statistics[::-1]
        # print(statistics)
        ublock_outputs = self.ublock_preconv(hidden_rep)
        # if torch.any(torch.isnan(ublock_outputs)):
        #     print('ublock_outputs before',torch.isnan(ublock_outputs))
        #     exit(0)
        for i, ublock in enumerate(self.ublocks):
            scale, shift = statistics[i]
            ublock_outputs = ublock(x=ublock_outputs, scale=scale, shift=shift)
            if torch.any(torch.isnan(scale)):
                input("scale")
                # input()
            if torch.any(torch.isnan(shift)):
                input("shift")
                # input()
            if torch.any(torch.isnan(ublock_outputs)):
                input("ublock_outputs")
                # input()
        outputs = self.ublock_postconv.forward(ublock_outputs)
        if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
            assert False, ("ublock_postconv")
        # print("outputs_shape", outputs.shape)
        return outputs.squeeze(1)

    def forward(self, yn, hidden_rep, noise_level):
        """
        Computes forward pass of neural network.
        :param mels (torch.Tensor): mel-spectrogram acoustic features of shape [B, n_mels, T//hop_length]
        :param yn (torch.Tensor): noised signal `y_n` of shape [B, T]
        :param noise_level (float): level of noise added by diffusion
        :return (torch.Tensor): epsilon noise
        """
        # Prepare inputs
        assert len(hidden_rep.shape) == 3  # B, n_mels, T
        # yn = yn.unsqueeze(1)
        
        if len(yn.shape) < 3:  # B, 1, T
            yn = yn.unsqueeze(1)
        print('ynshape',yn.shape)
        # Downsampling stream + Linear Modulation statistics calculation
        statistics = []
        dblock_outputs = self.dblock_preconv(yn)
        scale, shift = self.films[0](x=dblock_outputs, noise_level=noise_level)
        statistics.append([scale, shift])
        for dblock, film in zip(self.dblocks, self.films[1:]):
            dblock_outputs = dblock(dblock_outputs)
            scale, shift = film(x=dblock_outputs, noise_level=noise_level)
            statistics.append([scale, shift])
        statistics = statistics[::-1]

        # Upsampling stream
        ublock_outputs = self.ublock_preconv(hidden_rep)
        for i, ublock in enumerate(self.ublocks):
            scale, shift = statistics[i]
            ublock_outputs = ublock(x=ublock_outputs, scale=scale, shift=shift)
        outputs = self.ublock_postconv(ublock_outputs)
        return outputs.squeeze(1)


def dblock_ublock_recursive(self, step, current_db_output, noise_level,
                            hidden_rep):
    if (step >= len(self.dblocks)):
        return self.ublock_preconv(hidden_rep)
    else:
        dblock_outputs = self.dblocks[step].forward(current_db_output)
        scale, shift = self.films[step + 1].forward(
            x=dblock_outputs,
            noise_level=noise_level,
        )
        ublock_outputs = dblock_ublock_recursive(self, step + 1,
                                                 dblock_outputs, hidden_rep)
        return self.ublocks[len(self.dblocks) - 1 - step].forward(
            ublock_outputs, scale, shift)
