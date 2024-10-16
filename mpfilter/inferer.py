"""
Description: inferer
Author: Rainyl
Date: 2022-09-29 21:47:42
LastEditTime: 2023-01-24 10:20:19
"""

from argparse import ArgumentParser
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray
from onnxruntime import InferenceSession


class InferBase(object):
    _providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        ),
        ("CPUExecutionProvider",),
    ]
    _xx = np.arange(400, 4000, 1, dtype=np.float32)

    def __init__(
        self,
        model_path: str,
        conf_path: Union[str, None] = None,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.conf_path = conf_path
        self.device = device
        if device == "cpu":
            self.providers = self._providers[1]
        else:
            self.providers = self._providers
        # self.conf: ConfigBase

    def load_conf(self, conf_path: str = ""):
        raise NotImplementedError()

    def load_model(self):
        model = InferenceSession(
            self.model_path,
            providers=self.providers,
        )
        return model

    def minmax(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        return (x - x.min()) / (x.max() - x.min())

    def interpolate(
        self,
        xp: NDArray[np.float32],
        fp: NDArray[np.float32],
        x: Union[NDArray[np.float32], None] = None,
        left: float = 0,
        right: float = 0,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        assert xp.size == fp.size
        xp = np.sort(xp)
        # x should be 2D (B, N)
        # minmax
        fp1 = self.minmax(fp)
        # interpolate
        x = x or np.arange(400, 4000, 1, dtype=np.float32)
        yy = np.interp(x, xp, fp1, left=left, right=right)
        return (x, yy.astype(np.float32))

    def __call__(self, x: NDArray[np.float32]):
        raise NotImplementedError()


class InferClsBase(InferBase):
    def __init__(
        self,
        model_path: str,
        conf_path: Union[str, None] = None,
        model_name: str = "cnn1d",
        device: str = "cpu",
    ) -> None:
        super().__init__(model_path, conf_path, device)
        self.model_path = model_path
        self.model_name = model_name
        self.model: InferenceSession = self.load_model()

    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        # x should be (B, 3600) or (3600)
        assert x.ndim in (1, 2), f"the dimension of input is [{x.ndim}], not supported"
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        inputs = {self.model.get_inputs()[0].name: x}
        out = self.model.run(None, inputs)
        if self.model_name == "cnn1d":
            out = out[0]
        else:
            out = out[1]
        return out


class InferSeqBase(InferBase):
    def __init__(
        self,
        model_path: str,
        conf_path: Union[str, None] = None,
        device="cpu",
    ) -> None:
        super().__init__(model_path, conf_path=conf_path, device=device)
        self.model_path = model_path
        self.model: InferenceSession = self.load_model()

    def __call__(self, x: NDArray[np.float32]):
        # x should be (B, 3600) or (B, 1, 3600)
        assert x.ndim in (2, 3), f"the dimension of input is [{x.ndim}], not supported"
        if x.ndim == 2:
            x = np.expand_dims(x, 1)
        inputs = {self.model.get_inputs()[0].name: x}
        out = self.model.run(None, inputs)[0]
        out[out < 0] = 0
        return out


class InferVCNN(InferSeqBase):
    def __init__(
        self,
        model_path: str,
        conf_path: Union[str, None] = None,
        device="cpu",
    ) -> None:
        super().__init__(conf_path=conf_path, model_path=model_path, device=device)


class InferAE(InferSeqBase):
    def __init__(
        self,
        model_path: str,
        conf_path: Union[str, None] = None,
        device="cpu",
    ) -> None:
        super().__init__(conf_path=conf_path, model_path=model_path, device=device)


def infer_seq(x: NDArray[np.float32], device="cuda"):
    inferer = InferAE(
        "convert/ae/mp=-seed=0-epoch=48-step=1700-val_loss=0.001.onnx",
        device=device,
    )
    out = inferer(x)
    return out


def infer_cls(x: NDArray[np.float32], device="cpu"):
    inferer = InferClsBase(
        model_path="convert/cnn1d/0_0.2.onnx",
        device=device,
    )
    out = inferer(x)
    return out


def main():
    x = np.random.normal(size=(2, 3600)).astype(np.float32)
    # y = infer_seq(x)
    # print(y[0].shape)
    y = infer_cls(x)
    print(y.argmax(axis=1))


if __name__ == "__main__":
    main()
    # parser = ArgumentParser()
    # parser.add_argument("--config", dest="conf", type=str, required=True)

    # args = parser.parse_args(["--config", ""])
