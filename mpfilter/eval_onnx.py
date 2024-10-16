import glob
import json
import os
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

from .eval_sk import DataLoader, Dataset, save_topk
from .inferer import InferClsBase


def main(args: Namespace, path_ptn: str = r"\d{1,4}_" + f"{0.2}.onnx"):
    ds_name = Path(args.dataset).parent.name
    if args.mpe:
        ds_name = Path(args.dataset).name
    ds_name = ds_name.replace("filtered_", "").replace("ds_1k_", "").replace("ds_", "")
    model_name = Path(args.src).name
    paths = glob.glob(f"{args.src}/*.onnx")
    paths = [p for p in paths if re.search(path_ptn, p)]
    seeds_num = {
        "0": 0,
        "21": 1,
        "42": 2,
        "84": 3,
        "100": 4,
        "200": 5,
        "400": 6,
        "600": 7,
        "800": 8,
        "1000": 9,
    }
    dataset = Dataset(path=args.dataset)
    for p in paths:
        print(
            "====================================================\n"
            f"model: [{model_name}] at {p}, ds: [{args.dataset}]\n"
            "====================================================\n"
        )
        model = InferClsBase(model_path=p, device="cpu")
        y_true = []
        y_pred = []
        scores = []
        loader = DataLoader(ds=dataset, batch_size=args.bs)
        for yt, labels in tqdm(loader):
            yt = np.array(yt, dtype=np.float32)
            out = model(yt)
            preds = out.argmax(axis=1)
            y_true.extend(labels)
            y_pred.extend(preds.tolist())
            scores.append(out.tolist())

        p_name = Path(p).name
        seed = p_name.split("_")[0]
        num = num = seeds_num[seed]
        save_to = Path(f"{args.dst}/{model_name}/{num}/report_{ds_name}.json")
        if not save_to.parent.exists():
            save_to.parent.mkdir(parents=True)
        if args.mpe:
            args.saveto = save_to.parent / f"report_{ds_name}.csv"
            save_topk(
                scores,
                np.array(y_true),
                args,
            )
        else:
            report = metrics.classification_report(
                y_true=y_true,
                y_pred=y_pred,
                output_dict=True,
            )
            report["trues"] = y_true  # type: ignore
            report["preds"] = y_pred  # type: ignore
            with open(save_to, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", dest="src", type=str, required=True)
    parser.add_argument("-d", dest="dst", type=str, required=True)
    parser.add_argument("--mpe", dest="mpe", action="store_true")
    parser.add_argument("-bs", dest="bs", type=int, default=512)

    args = parser.parse_args(
        [
            "-s",
            "convert/cnn1d",
            "-d",
            "repeat1",
            # "repeat2",
            "-bs",
            "30",
            "--mpe",
        ]
    )
    all_ds = [
        # "G:\\mp_filter\\ds_1k_0.2\\val_test",
        # "G:\\mp_filter\\filtered_ae_0.2\\val_test",
        # "G:\\mp_filter\\filtered_conv_0.2\\val_test",
        # "/var/run/media/rainy/fx/dataset/mp_filter/ds_1k_0.2/val_test",
        # "/var/run/media/rainy/fx/dataset/mp_filter/filtered_ae_0.2/val_test",
        # "/var/run/media/rainy/fx/dataset/mp_filter/filtered_conv_0.2/val_test",
        # "data/dataset/ds_mpe",
        # "data/dataset/ds_mpe_ae",
        # "data/dataset/ds_mpe_conv",
        "data/dataset/ds_mpe_ori",
        "data/dataset/ds_mpe_ori_ae",
        "data/dataset/ds_mpe_ori_conv",
    ]
    all_dst = ["repeat1", "repeat2"]
    for ds in all_ds:
        args.dataset = ds
        args.model_name = Path(args.src).name
        for dst in all_dst:
            args.dst = dst
            if dst == "repeat1":
                ########### 02 only #######################
                main(args, path_ptn=r"\d{1,4}_" + f"{0.2}.onnx")
            else:
                ########### AE+VCNN+02 ####################
                main(args, path_ptn=r"\d{1,4}_" + r"02\+ae\+vcnn.onnx")
