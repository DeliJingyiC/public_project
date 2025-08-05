import subprocess
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List
import torch


def pad_list_dim2(
    tlist: List[torch.Tensor],
    tar_length: int,
    value: float = 0.0,
):
    padded = pad_sequence(
        tlist,
        batch_first=True,
        padding_value=value,
    )
    padded = F.pad(
        padded,
        pad=(0, tar_length - padded.shape[-1]),
        value=value,
    )

    return padded


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')
