"""
Description: config
Author: Rainyl
Date: 2022-09-09 15:56:59
LastEditTime: 2022-09-09 15:56:59
"""
import json
from pathlib import Path
from typing import Dict, List, Union


__supported_models__ = ("ae", "vcnn", "cnn1d")
__supported_device__ = ("cpu", "cuda")


class ConfigBase(object):
    def __init__(self, path: str, model: str) -> None:
        assert model in __supported_models__, f"model {self.model} not implemented"
        self.path: str = path
        self.model: str = model

    def load(self, p: Union[str, None] = None):
        p = p or self.path
        with open(p, "r", encoding="utf-8") as f:
            conf: Dict[str, Union[str, int, float]] = json.load(f)
        for k, v in conf.items():
            if getattr(self, k) is not None:
                setattr(self, k, v)

    @property
    def json(self):
        raise NotImplementedError()

    def save_init(self, p: Union[str, None] = None):
        p = p or self.path
        if not Path(p).parent.exists():
            Path(p).parent.mkdir(parents=True)
        d = self.json
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=4)


class ConfigCNN1D(ConfigBase):
    in_features: int = 1
    num_class: int = 25
    # common
    batch_size: int = 256
    dropout: float = 0.1
    num_tokens: int = 1000
    seq_len: int = 3600
    reserve: int = 3

    def __init__(self, path: str, model: str = "cnn1d", init=True) -> None:
        super().__init__(path, model)
        if init:
            self.save_init()
        else:
            self.load()

    @property
    def json(self):
        assert self.path, "config path is not set"
        d = {
            "in_features": self.in_features,
            "num_class": self.num_class,
            "dropout": self.dropout,
            "num_tokens": self.num_tokens,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "reserve": self.reserve,
        }
        return d


class ConfigVCNN(ConfigBase):
    in_features: int = 1
    out_features: int = 3600
    # common
    batch_size: int = 64
    dropout: float = 0.1
    num_tokens: int = 1000
    seq_len: int = 3600
    reserve: int = 3

    def __init__(self, path: str, model: str = "vcnn", init=True) -> None:
        super().__init__(path, model)
        if init:
            self.save_init()
        else:
            self.load()

    @property
    def json(self):
        assert self.path, "config path is not set"
        d = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "dropout": self.dropout,
            "num_tokens": self.num_tokens,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "reserve": self.reserve,
        }
        return d


class ConfigAE(ConfigBase):
    in_features: int = 3600
    hid_features: int = 128
    out_features: int = 3600

    # common
    batch_size: int = 64
    dropout: float = 0.1
    num_tokens: int = 1000
    seq_len: int = 3600
    reserve: int = 3

    def __init__(self, path: str, model: str = "ae", init=True) -> None:
        super().__init__(path, model)
        if init:
            self.save_init()
        else:
            self.load()

    @property
    def json(self):
        assert self.path, "config path is not set"
        d = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "hid_features": self.hid_features,
            "dropout": self.dropout,
            "num_tokens": self.num_tokens,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "reserve": self.reserve,
        }
        return d
