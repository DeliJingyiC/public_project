import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPredictor(nn.Module):

    def __init__(self, input_dim, channels):
        super().__init__()

        self.lstm = nn.LSTM(input_dim,
                            channels,
                            2,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(2 * channels, 1, bias=False), )
        # self.relu = nn.ReLU()

    def forward(self, x, mask, input_lengths):
        # x: [B, N, (chn.encoder + chn.speaker)]
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              input_lengths,
                                              batch_first=True,
                                              enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # [B, N, channels]
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.fc(x)  # [B, N, 1]
        # x = self.relu(x)
        x = x.squeeze(-1)  # [B, N]

        if mask is not None:
            x = x.masked_fill(mask, 0.0)

        return x

    def inference(self, x):
        # x: [B, N, (chn.encoder + chn.speaker)]
        # print("input", x)
        assert not torch.any(torch.isnan(x)), "org"

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # [B, N, channels]
        # print("lstm", x)
        assert not torch.any(torch.isnan(x)), "lstm"

        x = self.fc(x)  # [B, N, 1]
        # print("fc", x)
        # x = self.relu(x)
        assert not torch.any(torch.isnan(x)), "fc"

        x = x.squeeze(-1)  # [B, N]
        # print("squeeze", x)
        # input()
        assert not torch.any(torch.isnan(x)), "squeeze"
        # x = torch.relu(x)
        # assert torch.min(x) >= 0, x
        return x


class RangeParameterPredictor(nn.Module):

    def __init__(self, input_dim, channels):
        super().__init__()

        self.lstm = nn.LSTM(input_dim,
                            channels,
                            2,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * channels, 1, bias=False),
            nn.Softplus(),
        )

    def forward(self, x, duration, mask, input_lengths):
        # x: [B, N, (chn.encoder + chn.speaker)]
        # duration: [B, N]
        # print('x', x.shape)
        # assert False, f'x {x.shape,duration.unsqueeze(-1).shape} duration '
        x = torch.cat((x, duration.unsqueeze(-1)), dim=-1)
        # [B, N, (chn.encoder + chn.speaker) + 1]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              input_lengths,
                                              batch_first=True,
                                              enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # [B, N, channels]
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.fc(x)  # [B, N, 1]
        x = x.squeeze(-1)  # [B, N]

        if mask is not None:
            x = x.masked_fill(mask, 1e-8)

        return x

    def inference(self, x, duration):
        # x: [B, N, (chn.encoder + chn.speaker)]
        # duration: [B, N]
        x = torch.cat((x, duration.unsqueeze(-1)), dim=-1)
        # [B, N, (chn.encoder + chn.speaker) + 1]

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # [B, N, channels]

        x = self.fc(x)  # [B, N, 1]
        x = x.squeeze(-1)  # [B, N]

        return x


class GaussianUpsampling(nn.Module):

    def __init__(self):
        super().__init__()
        self.score_mask_value = 0.0

    def get_alignment_energies(self, gaussian, frames):
        energies = gaussian.log_prob(frames).exp()  # [B, N, T]
        return energies

    def forward(self, memory, duration, sigma, output_lengths, mask):
        print("forward", torch.max(output_lengths))
        # assert torch.max(output_lengths) > 0, f"{output_lengths}"
        frames = torch.arange(0,
                              torch.max(output_lengths),
                              device=memory.device)
        frames = frames.unsqueeze(0).unsqueeze(1)

        center = torch.cumsum(duration, dim=-1).float() - 0.5 * duration
        center, sigma = center.unsqueeze(-1), sigma.unsqueeze(-1)

        gaussian = torch.distributions.normal.Normal(loc=center, scale=sigma)

        alignment = self.get_alignment_energies(gaussian, frames)  # [B, N, T]

        if mask is not None:
            alignment = alignment.masked_fill(mask.unsqueeze(-1),
                                              self.score_mask_value)

        attn_weights = alignment / (
            torch.sum(alignment, dim=1).unsqueeze(1) + 1e-8)  # [B, N, T]
        upsampled = torch.bmm(attn_weights.transpose(1, 2), memory)
        # [B, T, N] @ [B, N, (chn.encoder + chn.speaker)] -> [B, T, (chn.encoder + chn.speaker)]

        return upsampled, attn_weights
