"""
Description: dataset generation
Author: Rainyl
Date: 2022-09-07 10:35:25
"""
import json
import os
import random
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import savgol_filter
from tqdm import tqdm

from .arg_parsers import DsGeneratorArgParser


def seed_everything(seed: int):
    # from kaggle, https://www.kaggle.com/code/rhythmcam/random-seed-everything
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def is_valid(y: NDArray[np.float32]):
    assert y.ndim == 1, f"y should be an 1-D array, but got {y.ndim}"
    rng = np.random.default_rng()
    return rng


def minmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
    is_valid(x)
    return (x - x.min()) / (x.max() - x.min())


def interpolate(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    xx: Union[NDArray[np.float32], None] = None,
    left: float = 0,
    right: float = 0,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    if xx is None:
        xx = np.arange(400, 4000, 1).astype(np.float32)
    assert x.ndim == y.ndim == 1 and x.size == y.size
    y1 = minmax(y)
    yy = np.interp(xx, x, y1, left=left, right=right)
    return (xx, yy.astype(np.float32))


# 0. no mutate
def no_mutate(y: NDArray[np.float32]) -> NDArray[np.float32]:
    return y


# 1. add noise
def add_noise(y1: NDArray[np.float32], max_std: float = 0.1) -> NDArray[np.float32]:
    rng = is_valid(y1)
    y = y1.copy()
    noise_std = rng.uniform(0, max_std)
    noise = rng.normal(0, noise_std, size=y.size)
    # noise = rng.uniform(0, 0.01, size=y.size)
    return y + noise


# 2. bend baseline
def bend_baseline(y1: NDArray[np.float32]) -> NDArray[np.float32]:
    rng = is_valid(y1)
    y = y1.copy()
    sign = -1 if rng.uniform() < 0.5 else 1
    n = rng.normal(1, 0.35, size=1)  # -0.05, 2.05
    s = rng.normal(0.1, 0.07, size=1)  # -0.11, 0.31
    xsin = minmax(np.arange(y.size)) * np.pi * n  # type: ignore
    ysin = sign * np.sin(xsin) * s
    yysin = ysin + minmax(y)
    return minmax(yysin)


# 3. random peaks
def add_rand_peaks(y1: NDArray[np.float32], step=314) -> NDArray[np.float32]:
    rng = is_valid(y1)
    y = y1.copy()
    level = rng.uniform(0.0, 0.6)
    num = rng.integers(1, 5)
    for _ in range(num):
        start_x = rng.integers(0, y.size - step)
        end_x = start_x + step
        x = np.arange(start_x, end_x).astype(np.float32)
        bend = minmax(np.sin(x / 200.0) + 1.0)
        y[start_x:end_x] = y[start_x:end_x] * (1 - level) + level * bend
    return y


# 4. reduce peak intensity
def reduce_peak(y1: NDArray[np.float32]) -> NDArray[np.float32]:
    rng = is_valid(y1)
    y = y1.copy()
    level = rng.uniform(0.5, 0.99)
    height = rng.uniform(0.1, 0.5)
    # peaks = find_peaks_cwt(spec.T[1], 100)
    # peaks, _ = find_peaks(spec.T[1], height=0.1, distance=1)
    peaks = minmax(y) > height
    # plt.figure(dpi=400)
    # plt.plot(spec.T[0], spec.T[1], lw=1)
    # plt.scatter(spec.T[0][peaks], spec.T[1][peaks], s=2, marker=".", c="g")
    y[peaks] = y[peaks] * level
    # plt.plot(spec.T[0], spec.T[1], lw=1, c="r")
    # plt.plot(spec.T[0], np.ones_like(spec.T[0]) * height, "--", c="k")
    # plt.savefig("reduce_peaks.jpg")
    return y


# preprocess, to remove or reduce the co2 data
def rm_co2(xx, y1: NDArray[np.float32], fac: float = 0.2):
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
    return y


def adj_negative(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    # x[x < 0] = 0
    # x = x + abs(x.min()) + 1 if x.min() < 1 else x
    x = x + abs(x.min()) if x.min() < 0 else x
    return x


def iModPolyFit(x, y, n: int):
    if not all([isinstance(x, np.ndarray), isinstance(y, np.ndarray)]):
        x, y = np.array(x), np.array(y)
    wavesOriginal = x
    devPrev = 0
    firstIter = True
    criteria = False
    while not criteria:
        paramVector = np.polynomial.Polynomial.fit(x, y, n)
        mod_poly = paramVector(x)
        residuals = y - mod_poly
        dev_curr = residuals.std(ddof=1)

        if firstIter:
            peaks = []
            for i in range(y.shape[0]):
                if y[i] > mod_poly[i] + dev_curr:
                    peaks.append(i)
            peaks = np.array(peaks)
            y = np.delete(y, peaks)
            mod_poly = np.delete(mod_poly, peaks)
            x = np.delete(x, peaks)
            firstIter = False
        for j in range(y.shape[0]):
            if mod_poly[j] + dev_curr > y[j]:
                pass
            else:
                y[j] = mod_poly[j]
        criteria = abs((dev_curr - devPrev) / dev_curr) <= 0.05

        if criteria:
            t = np.interp(wavesOriginal, x, y)
            return t.flatten()
        devPrev = dev_curr


def specy_prep(
    wavenum,
    absorb,
    smooth: int = 0,
    baseline: int = 0,
    adj_neg: bool = False,
):
    if adj_neg:
        absorb = adj_negative(absorb)
    if smooth == 0:
        smoothed = absorb
    else:
        smoothed: np.ndarray = savgol_filter(
            x=absorb,
            window_length=11,
            polyorder=smooth,
        )
    if baseline != 0:
        baselineRemove = smoothed - iModPolyFit(wavenum, smoothed, baseline)  # type: ignore
    else:
        baselineRemove = smoothed
    baselineRemove = minmax(baselineRemove)  # type: ignore
    return baselineRemove


def preprocess(df):
    return df


def save_one_file(xx_, yy_, yt_, save_path):
    assert len(xx_) == len(yy_) == len(yt_)
    tmp = np.vstack([xx_, minmax(yy_), minmax(yt_)]).T
    # tmp = (tmp * args.num_tokens + args.reserve).astype(np.int32)
    df = pd.DataFrame(data=tmp, columns=["x", "yy", "yt"])
    df.to_feather(save_path)


def generate(
    data: Dict[int, Dict[str, List]],
    args: DsGeneratorArgParser,
    range_min=400,
    range_max=4000,
):
    mutations: List[Callable[[NDArray[np.float32]], NDArray[np.float32]]] = [
        # no_mutate,
        add_noise,
        bend_baseline,
        add_rand_peaks,
        reduce_peak,
    ]

    rng = np.random.default_rng()
    XX = np.arange(range_min, range_max, 1).astype(np.float32)
    max_sample_per_mpid = args.max_num_per_id
    dataset = []
    full_ds = args.out_dir + "/full"
    if args.overwrite:
        if os.path.exists(full_ds):
            shutil.rmtree(full_ds)
    if not Path(full_ds).exists():
        Path(full_ds).mkdir(parents=True)

    data_keys = sorted(list(data.keys()))
    all_mpids = [(k, k) for k in data_keys]
    counts = {"+".join([str(m) for m in cm]): 0 for cm in all_mpids}
    label_hash = {i: list(m) for i, m in enumerate(all_mpids)}
    # generate label_name.json and save
    with open(full_ds + "/label_hash.json", "w") as f:
        json.dump(label_hash, f, indent=4)
    with open("data/mpid_type.json", "r") as f:
        mpid_label_hash = json.load(f)
    label_name = {}
    for k in label_hash:
        mpids = list(set(label_hash[k]))
        names = [mpid_label_hash[str(m)] for m in mpids]
        label_name[k] = names
    with open(full_ds + "/label_name.json", "w") as f:
        json.dump(label_name, f, indent=4)
    # start to generate dataset
    for label in tqdm(label_hash):
        # cm: mpids, eg. [0, 1]
        saveto = Path(full_ds + f"/{label}")
        cm = label_hash[label]
        count_key = "+".join([str(c) for c in cm])

        data_mpid_0 = data[cm[0]]
        data_mpid_1 = data[cm[1]]
        if not saveto.exists():
            saveto.mkdir(parents=True)
        len1 = len(data_mpid_0["x"])
        len2 = len(data_mpid_1["x"])
        label_done = False
        used_mix = []
        while not label_done:
            for ia in range(len1):
                if label_done:
                    break
                for ib in range(len2):
                    if label_done:
                        break
                    # get two spec randomly
                    # ia = rng.integers(0, len1)
                    # ib = rng.integers(0, len2)
                    xa, ya = np.asarray(data_mpid_0["x"][ia]), np.asarray(
                        data_mpid_0["y"][ia]
                    )
                    xb, yb = np.asarray(data_mpid_1["x"][ib]), np.asarray(
                        data_mpid_1["y"][ib]
                    )
                    # omit order by wavenum, done in sql
                    # idx = np.argsort(xa)
                    # xa, ya = xa[idx], ya[idx]
                    # interpolate to same wavenumbers
                    _, yya = interpolate(xa, ya, xx=XX)
                    yy = yya.copy()
                    if ia != ib:
                        _, yyb = interpolate(xb, yb, xx=XX)
                        yy = (yya + yyb) / 2
                    # perform openSpecy process methods accroding
                    # to a probablity
                    # if rng.uniform() < 0.4:
                    #     yy = specy_prep(XX, yy, smooth=3, baseline=8)
                    # perform co2 removing according to a probablity
                    if rng.uniform() < 0.7:
                        yy = rm_co2(XX, yy)

                    key_ab = "+".join([str(i) for i in {ia, ib}])
                    # limit max num of no mutate
                    nomutate_limit = counts[count_key] < max_sample_per_mpid * 0.2
                    if key_ab not in used_mix and nomutate_limit:
                        df_path = saveto / f"{counts[count_key]}.ftr"
                        save_one_file(XX, yy, yy, save_path=df_path)
                        dataset.append(df_path)
                        counts[count_key] += 1
                        used_mix.append(key_ab)
                    elif len(mutations) == 0:
                        ...
                    else:
                        n_mutate = rng.integers(0, len(mutations))
                        all_mutate_fns = combinations(mutations, n_mutate)
                        for mutate_fns in all_mutate_fns:
                            if counts[count_key] > max_sample_per_mpid:
                                break
                            yt = yy.copy()
                            for mutate_fn in mutate_fns:
                                if rng.uniform() < 0.5:
                                    yt = add_noise(
                                        yt, max_std=args.max_std
                                    )  # ensure noise added
                                yt = mutate_fn(yt)
                            df_path = saveto / f"{counts[count_key]}.ftr"
                            save_one_file(XX, yy, yt, save_path=df_path)
                            dataset.append(df_path)
                            counts[count_key] += 1
                    label_done = counts[count_key] >= max_sample_per_mpid
                    # if no any mutations performed, some label may not able to
                    # reach the max_per_mpid, the max number is len1 * len2
                    if len(mutations) == 0:
                        if cm[0] == cm[1]:
                            # for same type, the max number is
                            # $C_{len1}^2 + len1$
                            total = (len1 * (len1 - 1)) / (2 * 1) + len1
                            t = counts[count_key] >= total
                        else:
                            # for dirrerent type, the max number is len1 * len2
                            t = counts[count_key] >= len1 * len2
                        label_done = label_done or t

    if args.no_split:
        return "", "", ""

    dataset = np.asarray(dataset, dtype=str)
    # im_dataset = np.asarray(im_dataset, dtype=str)
    idxs = np.arange(len(dataset))
    rng.shuffle(idxs)
    dataset = dataset[idxs]
    # im_dataset = im_dataset[idxs]

    train_saveto = args.out_dir + "/train"
    val_saveto = args.out_dir + "/val"
    test_saveto = args.out_dir + "/test"
    for p in [train_saveto, val_saveto, test_saveto]:
        if os.path.exists(p):
            shutil.rmtree(p)
    copy_ds = [
        dataset,
    ]
    # if args.save_image:
    # copy_ds.append(im_dataset)
    for ds in copy_ds:
        n_all = len(ds)
        n_train = int(n_all * args.ptrain)
        n_val = int(n_all * args.pval)

        ds_train = ds[:n_train]
        ds_val = ds[n_train : n_train + n_val]
        ds_test = ds[n_train + n_val :]
        copy_to(ds_train, train_saveto)  # type: ignore
        copy_to(ds_val, val_saveto)  # type: ignore
        copy_to(ds_test, test_saveto)  # type: ignore

    return train_saveto, val_saveto, test_saveto


def copy_to(ds: List[str], dst: str):
    print(f"copying to {dst}")
    for d in tqdm(ds):
        dpath = Path(d)
        label, fname = dpath.parent.name, dpath.name
        saveto = Path(dst) / f"{label}" / f"{fname}"
        if not saveto.parent.exists():
            saveto.parent.mkdir(parents=True)
        shutil.copyfile(d, saveto)
        # print(f"{d} -> {saveto}")


def create_symlink(src, dst, is_dir=True):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if sys.platform == "win32":
        os.system(f"mklink {dst} {src}")
    elif sys.platform == "linux":
        os.system(f"ln -s {src} {dst}")
    else:
        raise NotImplementedError()


def main(args: DsGeneratorArgParser):
    if not Path(args.in_data).exists():
        raise ValueError(f"input date {args.in_data} not exist")
    if args.data_format == "feather":
        data = pd.read_feather(args.in_data)
        df = preprocess(data)
        mpids = df["mpid"].unique().astype(int)
        data_dict = {}
        # max_sample_per_id = 10  # for test
        max_sample_per_id = 1000
        for mpid in mpids:
            df_m = df[df["mpid"] == mpid]
            names = df_m["name"].unique()
            np.random.shuffle(names)
            names = names[:max_sample_per_id]
            data_list = [df_m[df_m["name"] == n] for n in names]
            tmp_dict = {
                "x": [d["wavenum"].tolist() for d in data_list],
                "y": [d["intensity"].tolist() for d in data_list],
            }
            data_dict[int(mpid)] = tmp_dict
        data_dict_save = args.out_dir + "/data_dict.json"
        if Path(data_dict_save).exists():
            if args.overwrite:
                with open(data_dict_save, "w") as f:
                    json.dump(data_dict, f, indent=4)
        else:
            with open(data_dict_save, "w") as f:
                json.dump(data_dict, f, indent=4)
        train_save, val_save, test_save = generate(data=data_dict, args=args)
        if (not args.no_split) and args.merge_val_test:
            print("making symbol links...")
            val_test_dir = Path(args.out_dir) / "val_test"
            if not val_test_dir.exists():
                val_test_dir.mkdir(parents=True)
            create_symlink(val_save, f"{val_test_dir / 'val'}")
            print(f"{val_save} -> {val_test_dir / 'val'}")
            create_symlink(test_save, f"{val_test_dir / 'test'}")
            print(f"{test_save} -> {val_test_dir / 'test'}")
    else:
        raise NotImplementedError(f"data format {args.data_format} not supported")


if __name__ == "__main__":
    parser = DsGeneratorArgParser()

    args: DsGeneratorArgParser = parser.parse_args()

    if not os.path.exists(args.out_dir):
        Path(args.out_dir).mkdir(parents=True)

    seed_everything(42)

    main(args)
