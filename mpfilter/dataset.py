"""
Description: dataset
Author: Rainyl
Date: 2022-09-08 09:25:00
"""
import os
import glob
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MpDataset(Dataset):
    def __init__(self, path: str, fmt="ftr", transforms=None) -> None:
        super(MpDataset, self).__init__()
        self.path = path
        if os.path.islink(path):
            self.path = os.path.realpath(path)
        assert os.path.exists(
            self.path
        ), f"dataset path {path} not exist or link broken"
        self.fmt = fmt
        self.transforms = transforms
        self.data_path = glob.glob(self.path + f"/**/*.{self.fmt}", recursive=True)

        self._len: int = len(self.data_path)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        raise NotImplementedError()


class MpDatasetCls(MpDataset):
    def __init__(
        self,
        path: str,
        fmt="ftr",
        transforms=None,
        num_token: int = 1000,
        reserve: int = 3,
        seq_len: int = 3600,
    ) -> None:
        super(MpDatasetCls, self).__init__(path, fmt, transforms)
        self.num_token = num_token
        self.reserve = reserve
        self.seq_len = seq_len
        paths = [p.split(os.sep)[-2] for p in self.data_path]
        self.labels = [int(p) for p in paths]

    def __getitem__(self, idx: int):
        assert idx < self._len, f"index {idx} out of range {self._len}"
        dpath = self.data_path[idx]
        spec: pd.DataFrame = pd.read_feather(dpath)
        # x, yy, yt
        x, yy, yt = spec.values.T
        assert (
            yt.shape[0] <= self.seq_len
        ), f"data length {yt.shape[0]} overflow!, file {self.data_path[idx]}"
        yt = torch.from_numpy(yt)
        if self.transforms:
            yt = self.transforms(yt.unsqueeze(0))[0]
        label = self.labels[idx]
        return yt, label


class MpDatasetSK(MpDatasetCls):
    def __getitem__(self, idx: int):
        assert idx < self._len, f"index {idx} out of range {self._len}"
        dpath = self.data_path[idx]
        spec: pd.DataFrame = pd.read_feather(dpath)
        # x, yy, yt
        x, yy, yt = spec.values.T
        label = self.labels[idx]
        return yt, label


class MpDatasetSeq(MpDataset):
    def __init__(
        self,
        path: str,
        fmt: str = "ftr",
        transforms=None,
        seq_len: int = 3600,
    ) -> None:
        super(MpDatasetSeq, self).__init__(path, fmt, transforms)
        self.seq_len = seq_len

    def __getitem__(self, idx):
        assert idx < self._len, f"index {idx} out of range {self._len}"
        spec: pd.DataFrame = pd.read_feather(self.data_path[idx])
        x, yy, yt = spec.values.T
        assert (
            yt.shape[0] <= self.seq_len
        ), f"data length {yt.shape[0]} overflow!, file {self.data_path[idx]}"
        yy = torch.from_numpy(yy.astype(np.float32))
        yt = torch.from_numpy(yt.astype(np.float32))
        # seq_aug, seq_true
        return yt, yy


def collate_fn_seq(batch):
    src: torch.Tensor = pad_sequence(
        [b[0] for b in batch],
        batch_first=True,
        padding_value=0,
    ).to(torch.int32)
    tgt: torch.Tensor = pad_sequence(
        [b[1] for b in batch],
        batch_first=True,
        padding_value=0,
    ).to(torch.int32)
    return [src, tgt]
