"""
Transform dataset using constructed AE and VCNN
"""

import glob
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from .inferer import InferAE, InferSeqBase, InferVCNN


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
        self._len: int = len(self.data_path)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        assert idx < self._len, f"index {idx} out of range {self._len}"
        dpath = self.data_path[idx]
        spec: pd.DataFrame = pd.read_feather(dpath)
        # x, yy, yt
        x, yy, yt = spec.values.T
        return x, yy, yt, dpath


class DataLoader(object):
    def __init__(self, ds: Dataset, batch_size: int = 1) -> None:
        self.ds = ds
        self.batch_size = batch_size
        self.start = 0
        self.stop = self.start + batch_size
        self.max_len = len(ds)

    def __len__(self):
        l = self.max_len / self.batch_size
        return l + 1 if l - int(l) > 0 else int(l)

    def __next__(self):
        if self.start == self.stop:
            raise StopIteration
        res = [self.ds[i] for i in range(self.start, self.stop)]
        x = np.asarray([r[0] for r in res], dtype=np.float32)
        yy = np.asarray([r[1] for r in res], dtype=np.float32)
        yt = np.asarray([r[2] for r in res], dtype=np.float32)
        p = [r[3] for r in res]
        self.start = self.stop
        self.stop += self.batch_size
        self.stop = min(self.stop, self.max_len)
        return x, yy, yt, p

    def __iter__(self):
        return self


def main(args: Namespace):
    if args.model == "ae":
        inferer = InferAE(args.model_path, device=args.device)
    else:
        inferer = InferVCNN(args.model_path, device=args.device)
    if not Path(args.dst_ds).exists():
        Path(args.dst_ds).mkdir(parents=True)

    ds = Dataset(args.src_ds)
    loader = DataLoader(ds=ds, batch_size=args.batch_size)
    all_out = []
    for x, yy, yt, p in iter(tqdm(loader)):
        out = inferer(yt)
        tmp = {"x": x, "yy": yy, "yt": out, "p": p}
        all_out.append(tmp)
    for d in tqdm(all_out):
        for i in range(len(d["p"])):
            m = np.vstack((d["x"][i], d["yy"][i], d["yt"][i])).T
            df = pd.DataFrame(m, columns=["x", "yy", "yt"])
            pp = Path(d["p"][i])
            save_to_dir = Path(args.dst_ds) / pp.parent.parent.name / pp.parent.name
            if not save_to_dir.exists():
                save_to_dir.mkdir(parents=True)
            save_to = save_to_dir / pp.name
            df.to_feather(save_to)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src-ds", dest="src_ds", type=str, required=True)
    parser.add_argument("--dst-ds", dest="dst_ds", type=str, required=True)
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        required=True,
        choices=["ae", "vcnn"],
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        required=False,
        default="cuda",
        choices=["cpu", "cuda"],
    )

    # parser.add_argument("--model-conf", dest="model_conf", type=str, required=True)
    parser.add_argument("--model-path", dest="model_path", type=str, required=True)

    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)

    args = parser.parse_args()
    main(args)
