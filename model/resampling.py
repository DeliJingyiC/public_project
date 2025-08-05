import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_upsampling import DurationPredictor, RangeParameterPredictor, GaussianUpsampling


class Resampling(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.duration_predictor = DurationPredictor(
            hparams.encoder.channel + hparams.encoder.speaker_emb,
            hparams.dur_predictor.dur_lstm_channel)
        self.range_parameter_predictor = RangeParameterPredictor(
            hparams.encoder.channel + hparams.encoder.speaker_emb + 1,
            hparams.dur_predictor.range_lstm_channel)
        self.gaussian_upsampling = GaussianUpsampling()

    def forward(
        self,
        memory,
        target_duration,
        memory_lengths,
        output_lengths,
        no_mask=False,
    ):
        mask = None if no_mask else ~self.get_mask_from_lengths(memory_lengths)
        # assert False, f'mask {len(mask)}'

        duration_s = self.duration_predictor(
            memory,
            mask,
            memory_lengths,
        )  # [B, N]
        # print("line 37resampling", memory.shape)
        # print("line 38resampling", target_duration.shape)
        # assert False, f'duration_s{duration_s.shape}'
        # input([
        #     x.dtype for x in [
        #         memory,
        #         target_duration,
        #         mask,
        #         memory_lengths,
        #     ]
        # ])
        sigma = self.range_parameter_predictor(
            memory,
            target_duration,
            mask,
            memory_lengths,
        )  # [B, N]
        # print("line 45resampling", target_duration.shape)

        upsampled, alignments = self.gaussian_upsampling(
            memory,
            target_duration,
            sigma,
            output_lengths,
            mask,
        )
        # upsampled: [B, T, (chn.encoder + chn.speaker)], alignments: [B, N, T]
        # print("line 55resampling", target_duration.shape)

        return upsampled.transpose(1, 2), alignments, duration_s, mask

    def inference(self, memory, pace):
        duration_s = self.duration_predictor.inference(memory)  # [B, N]
        # print("duration_sline63",duration_s.size())
        duration_s = duration_s * pace
        # print("duration_sline65",duration_s.size())

        # input(duration_s)
        duration = torch.round(
            duration_s *
            (self.hparams.audio.sampling_rate / self.hparams.window.scale))
        # print("duration_line71",duration.size())

        # input(torch.min(duration))
        # input(torch.max(duration))

        sigma = self.range_parameter_predictor.inference(
            memory,
            duration,
        )
        # print("sigma",sigma.size())
        # [B, N]
        # print(duration)
        # print(torch.sum(duration, dim=-1).detach().shape)
        output_len = int(torch.sum(duration, dim=1).detach())
        # print("output_len",output_len)
        # print("pass")
        # print("output_len", output_len)
        # assert False
        # input(torch.min(duration))
        # input(torch.min(output_len))
        # input(torch.max(output_len))

        upsampled, alignments = self.gaussian_upsampling.forward(
            memory,
            duration,
            sigma,
            torch.tensor([output_len]),
            # output_len,
            mask=None,
        )
        # print("upsampled",upsampled.size())
        # print("upsampledtrans",upsampled.transpose(1, 2).size())
        # assert False
        # upsampled: [B, T, (chn.encoder + chn.speaker)], alignments: [B, N, T]
        if np.prod(upsampled.transpose(1, 2).size()) == 0:
            print("upsampled.transpose(1, 2).size()",
                  upsampled.transpose(1, 2).size())
            print("duration", duration.size())
            print("upsampled", upsampled.size())
            print("output_len", output_len)
        return upsampled.transpose(1, 2), alignments, duration, sigma

    def inference_parallel(self, memory, pace):
        duration_s = self.duration_predictor.inference(memory)  # [B, N]
        duration_s = duration_s * pace
        duration = torch.round(
            duration_s *
            (self.hparams.audio.sampling_rate / self.hparams.window.scale))

        sigma = self.range_parameter_predictor.inference(memory,
                                                         duration)  # [B, N]
        # assert torch.max(
        #     duration) > 0, f"inference_parallel duration {duration}"
        output_len = torch.sum(duration, dim=-1).detach()
        output_len_int = output_len.to(torch.int32)
        # output_len_int = torch.relu(output_len_int)
        assert torch.max(
            output_len_int) > 0, f"inference_parallel {output_len_int}"
        upsampled, alignments = self.gaussian_upsampling.forward(
            memory,
            duration,
            sigma,
            output_len_int,
            mask=None,
        )
        # upsampled: [B, T, (chn.encoder + chn.speaker)], alignments: [B, N, T]

        return upsampled.transpose(1, 2), alignments, duration, sigma

    def get_mask_from_lengths(self, lengths, max_len=None):
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(
            0,
            max_len,
        ).type_as(lengths)
        mask = (ids < lengths.unsqueeze(1))
        return mask
