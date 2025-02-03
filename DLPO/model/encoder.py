import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class TextEncoder(nn.Module):

    def __init__(self, channels, kernel_size, depth, dropout_rate, n_symbols):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = list()
        for _ in range(depth):
            self.cnn.append(
                nn.Sequential(
                    nn.Conv1d(channels,
                              channels,
                              kernel_size=kernel_size,
                              padding=padding),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                    # nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate),
                ))
        self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(
            channels,
            channels // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        x = self.cnn(x)  # [B, chn, T]
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        # input(x.shape)
        # input(input_lengths.shape)
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              input_lengths,
                                              batch_first=True,
                                              enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        # input(x.shape)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        #x = F.dropout(x, self.dropout_rate, self.training)

        return x

    def inference(self, x):
        assert not torch.any(torch.isnan(x)), "x1"
        # print("before", x)
        for n, p in self.embedding.named_parameters():
            assert not torch.any(torch.isnan(p)), n
        x = self.embedding.forward(x)

        assert not torch.any(torch.isnan(x)), x
        x = x.transpose(1, 2)
        assert not torch.any(torch.isnan(x)), "x3"
        x = self.cnn(x)
        assert not torch.any(torch.isnan(x)), "x4"
        x = x.transpose(1, 2)
        assert not torch.any(torch.isnan(x)), "x5"
        self.lstm.flatten_parameters()
        assert not torch.any(torch.isnan(x)), "x6"
        x, _ = self.lstm(x)
        return x
