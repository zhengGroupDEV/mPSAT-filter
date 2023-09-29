"""
Description: convert CNN to ONNX format
Author: Rainyl
License: Apache License 2.0
"""
import os
import random
import re
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from .config import ConfigAe, ConfigCconv, ConfigConv
from .model import (
    MpModelClsConv,
    MpModelSeqAe,
    MpModelSeqConv,
)


def seed_everything(seed: int):
    print(f"set global seed to {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def replace_state_dict(ckpt: str, device: str):
    sd_new = OrderedDict()
    pl_state_dict = torch.load(ckpt, map_location=device)

    sd1 = pl_state_dict["state_dict"]
    for k in sd1:
        k1 = k.replace("model.", "")
        sd_new[k1] = sd1[k]
    return sd_new


def to_onnx(ckpt: str, save_to: str, model_name: str):
    device = "cpu"
    providers = ("CPUExecutionProvider",)
    seed_everything(42)
    if model_name == "cconv":
        # CNN
        conf = ConfigCconv("config/config_cconv.json")
        model = MpModelClsConv(
            in_features=conf.in_features,
            out_features=conf.num_class,
            dropout=conf.dropout,
        )
        x = torch.randn(1, 3600)
    elif model_name == "conv":
        conf = ConfigConv("config/config_conv.json", init=False)
        model = MpModelSeqConv(
            in_features=conf.in_features,
            out_features=conf.out_features,
            dropout=conf.dropout,
        )
        x = torch.randn(1, 1, 3600)
    elif model_name == "ae":
        conf = ConfigAe("config/config_ae.json")
        model = MpModelSeqAe(
            in_features=conf.in_features,
            hid_features=conf.hid_features,
            out_features=conf.out_features,
            dropout=conf.dropout,
        )
        x = torch.randn(1, 1, 3600)
    else:
        raise ValueError()
    state_dict = replace_state_dict(ckpt, device)
    torch.save(state_dict, save_to.replace(".onnx", ".pth"))
    model.load_state_dict(state_dict)
    model.eval()
    out = model(x)

    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model,
        x,
        save_to,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    onnx_model = onnx.load(save_to)
    onnx.checker.check_model(onnx_model)  # type: ignore
    ort_session = onnxruntime.InferenceSession(save_to, providers=providers)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print(f"Success! {ckpt} -> {save_to}")


def main(args: Namespace):
    src = Path(args.src)
    dst = Path(args.dst) / src.name
    if not dst.exists():
        dst.mkdir(parents=True)

    ckpts = src.glob("**/*.ckpt")
    for p in ckpts:
        src_path = str(p)
        ckpt_name = re.search(r"seed=\d{1,4}.+-epoch", p.name)
        if ckpt_name:
            ckpt_name = (
                ckpt_name.group()
                .replace("seed=", "")
                .replace("-1024", "")
                .replace("-epoch", "")
                + ".onnx"
            )
        else:
            ckpt_name = p.name.replace(".ckpt", ".onnx")
        dst_path = dst / ckpt_name

        to_onnx(ckpt=src_path, save_to=str(dst_path), model_name="cconv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", dest="src", type=str, required=True)
    parser.add_argument("-d", dest="dst", type=str, required=True)

    args = parser.parse_args()
    main(args)
