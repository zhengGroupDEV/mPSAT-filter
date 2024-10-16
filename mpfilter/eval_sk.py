import glob
import json
import os
import pickle
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn import metrics
from tqdm import tqdm


class Dataset(object):
    def __init__(self, path: str, fmt="ftr") -> None:
        super(Dataset, self).__init__()
        self.path = path
        if os.path.islink(path):
            self.path = os.path.realpath(path)
        assert os.path.exists(
            self.path
        ), f"dataset path {path} not exist or link broken"
        self.fmt = fmt
        self.data_path = glob.glob(self.path + f"/**/*.{self.fmt}", recursive=True)
        self.labels = [int(Path(p).parent.name) for p in self.data_path]
        self._len: int = len(self.data_path)

    def __len__(self) -> int:
        return self._len

    def minmax(self, x, axis=0):
        return (x - x.min()) / (x.max() - x.min())

    def rm_co2(
        self, xx: NDArray[np.float32], y1: NDArray[np.float32], fac: float = 0.2
    ):
        y = y1.copy()
        c1 = (xx >= 2280) & (xx <= 2400)
        c2 = (xx >= 3570) & (xx <= 3750)
        # replace with a line
        min1 = min(y[c1])
        min2 = min(y[c2])
        y[c1] = min1
        y[c2] = min2
        # scale by fac
        # idxs = (c1 | c2)
        # y[idxs] = y[idxs] * fac
        y[y < 0] = 0
        return self.minmax(y)

    def __getitem__(self, idx: int):
        assert idx < self._len, f"index {idx} out of range {self._len}"
        dpath = self.data_path[idx]
        spec: pd.DataFrame = pd.read_feather(dpath)
        label = self.labels[idx]
        # x, yy, yt
        x, yy, yt = spec.values.astype(np.float32).T
        yt = self.rm_co2(x, yt)
        return yt, label


class DataLoader(object):
    def __init__(self, ds: Dataset, batch_size: int = 1) -> None:
        self.ds = ds
        self.batch_size = batch_size
        self.start = 0
        self.max_len = len(ds)
        self.stop = min(self.max_len, self.start + batch_size)

    def __len__(self):
        len_ = self.max_len / self.batch_size
        return len_ + 1 if len_ - int(len_) > 0 else int(len_)

    def __next__(self):
        if self.start == self.stop:
            raise StopIteration
        res = [self.ds[i] for i in range(self.start, self.stop)]
        yt = [r[0] for r in res]
        p = [r[1] for r in res]
        self.start = self.stop
        self.stop += self.batch_size
        self.stop = min(self.stop, self.max_len)
        return yt, p

    def __iter__(self):
        return self


def softmax(x: NDArray[np.float32], axis=1) -> NDArray[np.float32]:
    # x: (B, num_class)
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def save_topk(
    scores_: List[NDArray[np.float32]],
    y_trues: NDArray[np.float32],
    args: Namespace,
    # names: NDArray[np.float32],
):
    LABEL_NAME = {
        "0": "HDPE",
        "1": "LDPE",
        "2": "LLDPE",
        "3": "PET",
        "4": "ABS",
        "5": "PP",
        "6": "PS",
        "7": "PVC",
        "8": "CA",
        "9": "PMMA",
        "10": "PA",
        "11": "PC",
        "12": "PLA",
        "13": "PBT",
        "14": "CPE",
        "15": "EVA",
        "16": "PU",
        "17": "PTFE",
        "18": "POM",
        "19": "PVDF",
        "20": "PCL",
        "21": "EVOH",
        "22": "PVOH",
        "23": "PVA",
        "24": "UNKNOWN",
    }
    scores: NDArray[np.float32] = np.concatenate(scores_, axis=0)
    scores = softmax(scores, axis=1)
    names = np.arange(scores.shape[0])
    if names.ndim == 1:
        names = np.expand_dims(names, 1)
    args.topk = 10
    args.num_class = 25

    acc = metrics.top_k_accuracy_score(
        y_trues,
        scores,
        k=args.topk,
        labels=np.arange(args.num_class),
    )
    print(
        f"############# result of eval mpe ############\n"
        f"top-{args.topk} Accuracy: {acc}\n"
    )
    save_to = Path(args.saveto)
    y_trues_name = np.array(
        [LABEL_NAME[str(y)] for y in y_trues],
        dtype=str,
    )
    out_topk = scores.argsort(axis=1)[:, ::-1][:, : args.topk]
    out_topk = np.array([[LABEL_NAME[str(y)] for y in r] for r in out_topk], dtype=str)
    notes = ["label"] * out_topk.shape[0]
    out_topk = np.vstack((y_trues_name, out_topk.T, notes)).T

    out_scores = scores.copy()
    out_scores.sort(axis=1)
    out_scores = out_scores[:, ::-1][:, : args.topk]
    notes = ["score"] * out_topk.shape[0]
    out_scores = np.vstack((y_trues_name, out_scores.T, notes), dtype=str).T

    out_df_list = []
    assert out_topk.shape == out_scores.shape
    for i in range(out_topk.shape[0]):
        out_df_list.append(out_topk[i])
        out_df_list.append(out_scores[i])
    out_df = pd.DataFrame(
        data=out_df_list,
        dtype=str,
        columns=["true", *[f"pred{i}" for i in range(args.topk)], "note"],
    )
    out_df.to_csv(save_to, index=False)
    print(f"Saved eval results to {save_to}")
    return acc


def load_clf_pkl(pkl: str):
    print(f"Loading pkl [{pkl}] ...")
    if not os.path.exists(pkl):
        raise ValueError(f"pkl {pkl} not exists")
    with open(pkl, "rb") as f:
        clf = pickle.load(f)
    return clf


def main(args: Namespace, path_ptn: str = r"\d{1,4}_" + f"{0.2}.onnx"):
    ds_name = Path(args.dataset).parent.name
    if args.mpe:
        ds_name = Path(args.dataset).name
    ds_name = ds_name.replace("filtered_", "").replace("ds_1k_", "").replace("ds_", "")
    model_name = Path(args.src).name
    pkl_paths = glob.glob(f"{args.src}/**/*.pkl")
    pkl_paths = [p for p in pkl_paths if re.search(path_ptn, p)]
    dataset = Dataset(path=args.dataset)
    for p in pkl_paths:
        print(
            "====================================================\n"
            f"model: [{model_name}] at {p}, ds: [{args.dataset}]\n"
            "====================================================\n"
        )
        model = load_clf_pkl(p)
        y_true = []
        y_pred = []
        scores = []
        loader = DataLoader(ds=dataset, batch_size=args.bs)
        for yt, labels in tqdm(loader):
            yt = np.array(yt, dtype=np.float32)
            if model_name in ("dt", "rf"):
                out: NDArray[np.float32] = model.predict_proba(yt)
            else:
                out: NDArray[np.float32] = model.decision_function(yt)
            preds = out.argmax(axis=1)
            y_true.extend(labels)
            y_pred.extend(preds.tolist())
            scores.append(out.tolist())
        num = Path(p).parent.name
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
    parser.add_argument("-bs", dest="bs", type=int, default=512)
    parser.add_argument("--mpe", dest="mpe", action="store_true")

    args = parser.parse_args(
        [
            # "-s",
            # "repeat/dt",
            # # "repeat/rf",
            # # "repeat/lsvm",
            # "-d",
            # "repeat2",
            "-s",
            # "repeat1/dt",
            "repeat1/rf",
            # "repeat1/lsvm",
            "-d",
            "repeat1",
            # "-bs", "30",
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
    all_repeat = ["repeat", "repeat1"]
    all_model_names = ["dt", "rf", "lsvm"]
    for ds in all_ds:
        args.dataset = ds
        for rep in all_repeat:
            for mn in all_model_names:
                args.src = f"{rep}/{mn}"
                args.model_name = Path(args.src).name
                ########### 02 only #######################
                if rep == "repeat":
                    args.dst = "repeat2"
                    main(args, path_ptn=rf"clf_{args.model_name}_0.2.pkl")
                ########### AE+VCNN+02 ####################
                else:
                    args.dst = "repeat1"
                    main(args, path_ptn=r"02\+ae\+vcnn.pkl")
