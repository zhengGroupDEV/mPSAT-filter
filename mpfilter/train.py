import os
import random
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import seed_everything
from torch.utils.data import DataLoader

# from .arghint import ArgsTrain
from .config import (
    ConfigAe,
    ConfigCconv,
    ConfigConv,
    __supported_device__,
    __supported_models__,
)
from .dataset import MpDatasetCls, MpDatasetSeq

# from .transforms import Fil
from .inferer_nn import InferNN
from .model_pl import (
    MpModelAeLight,
    MpModelCconvLight,
    MpModelConvLight,
    MpSaveCkptOnShutdown,
)


def load_data_set(
    path: str,
    args: Namespace,
    num_tokens: int = 1000,
    seq_len: int = 3600,
    shuffle=True,
) -> DataLoader:
    model = args.model
    assert model in __supported_models__
    if model in ("ae", "conv"):
        loader = DataLoader(
            dataset=MpDatasetSeq(
                path=path,
                fmt="ftr",
                transforms=None,
                seq_len=seq_len,
            ),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            # collate_fn=collate_fn_seq,
        )
    elif model in ("cconv",):
        ds_transform = None
        if args.transform:
            assert (
                args.transform and args.transform_ckpt
            ), "must provide transform_ckpt if use transform"
            print(f"Using [{args.transform}] at {args.transform_ckpt} as transform")
            ds_transform = InferNN(
                args.transform_ckpt,
                conf_path=args.transform_conf,
                model_name=args.transform,
                device=args.device,
            )
        loader = DataLoader(
            dataset=MpDatasetCls(
                path=path,
                fmt="ftr",
                transforms=ds_transform,
                num_token=num_tokens,
                seq_len=seq_len,
            ),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        raise ValueError(f"model {model} not supported")
    return loader


def get_model_conf(args: Namespace):
    # config
    init = args.overwrite_config
    if args.model == "ae":
        conf = ConfigAe(args.conf, model=args.model, init=init)
        model = MpModelAeLight(
            conf=conf,
            optimizer=args.optim,
            lr=args.lr,
            sc_step=args.sc_step,
            sc_gamma=args.sc_gamma,
            warmup_step=args.warmup,
        )
    elif args.model == "conv":
        conf = ConfigConv(args.conf, model=args.model, init=init)
        model = MpModelConvLight(
            conf=conf,
            optimizer=args.optim,
            lr=args.lr,
            sc_step=args.sc_step,
            sc_gamma=args.sc_gamma,
            warmup_step=args.warmup,
        )
    elif args.model == "cconv":
        conf = ConfigCconv(args.conf, model=args.model, init=init)
        model = MpModelCconvLight(
            conf=conf,
            optimizer=args.optim,
            lr=args.lr,
            sc_step=args.sc_step,
            sc_gamma=args.sc_gamma,
            warmup_step=args.warmup,
        )
    else:
        raise ValueError()

    return model, conf


def main(args: Namespace):
    print(
        f"Training dataset: {args.train_set}\n"
        f"Validation dataset: {args.val_set}\n"
        f"Test dataset: {args.test_set}\n"
    )
    model, conf = get_model_conf(args)
    args.batch_size = args.batch_size or conf.batch_size

    # logger
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    version_info = (
        f"model_{args.model}-lr_{args.lr}-bs_{args.batch_size}"
        f"-seed_{args.seed}-warmup_{args.warmup}"
        f"-extra_{args.extra}-time_{time_stamp}"
    )
    logger = TensorBoardLogger(
        save_dir="logs",
        name=args.model,
        # log_graph=True,
        version=version_info,
    )

    # callbacks
    # dev_monitor = DeviceStatsMonitor()
    save_on_shutdown = MpSaveCkptOnShutdown()
    grad_accum = GradientAccumulationScheduler({0: args.accum_step})
    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_name = (
        f"mp_seed={args.seed}-{args.extra}"
        "-epoch={epoch}-step={step}-val_loss={val/loss:.3f}"
    )
    time_now = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    ckpt_callback = ModelCheckpoint(
        dirpath=f"ckpts/{args.model}/{time_now}",
        filename=ckpt_name,
        # monitor="val/loss",
        # mode="min",
        every_n_train_steps=args.save_step,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        save_weights_only=True,
    )
    early_stop = EarlyStopping(
        monitor="val/loss",
        min_delta=0,
        patience=args.es_patience,
        mode="min",
    )

    # trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            # dev_monitor,
            grad_accum,
            lr_monitor,
            ckpt_callback,
            save_on_shutdown,
            early_stop,
        ],
        accelerator=args.device,
        devices=1,
        gradient_clip_val=1,
        max_epochs=args.epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        # val_check_interval=0.5,
        val_check_interval=None,
        precision=32,
        num_sanity_val_steps=1,
        # strategy="ddp",
    )

    # dataset and dataloader
    train_loader = load_data_set(
        path=args.train_set,
        args=args,
        num_tokens=conf.num_tokens,
        seq_len=conf.seq_len,
        shuffle=True,
    )
    val_loader = load_data_set(
        path=args.val_set,
        args=args,
        num_tokens=conf.num_tokens,
        seq_len=conf.seq_len,
        shuffle=False,
    )

    test_loader = load_data_set(
        path=args.test_set,
        args=args,
        num_tokens=conf.num_tokens,
        seq_len=conf.seq_len,
        shuffle=False,
    )

    # lr_finder = trainer.tuner.lr_find(model)

    # trainer.tune(
    #     model=model,
    #     train_dataloaders=train_loader,
    #     val_dataloaders=val_loader,
    #     lr_find_kwargs={"min_lr": 1e-5, "max_lr": 0.01},
    # )

    # fit
    ckpt_path = "ckpts/interrupt.ckpt" if args.use_resume else None
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

    trainer.test(dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train-set", dest="train_set", type=str, required=True)
    parser.add_argument("--test-set", dest="test_set", type=str, required=True)
    parser.add_argument("--val-set", dest="val_set", type=str, required=True)
    parser.add_argument("--config", dest="conf", type=str, required=True)
    parser.add_argument(
        "--overwrite-config",
        dest="overwrite_config",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
    )
    parser.add_argument(
        "--use-resume",
        dest="use_resume",
        action="store_true",
    )

    parser.add_argument("--model", dest="model", type=str, default="cls")
    parser.add_argument("--device", dest="device", type=str, default="cpu")
    parser.add_argument("--transform", dest="transform", type=str, default="ae")
    parser.add_argument("--transform-ckpt", dest="transform_ckpt", type=str, default="")
    parser.add_argument("--transform-conf", dest="transform_conf", type=str, default="")
    parser.add_argument("--optim", dest="optim", type=str, default="adam")
    parser.add_argument("--epochs", dest="epochs", type=int, default=30)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--lr", dest="lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", dest="batch_size", type=int, required=False)

    parser.add_argument("--save-step", dest="save_step", type=int, default=2)
    parser.add_argument("--es-patience", dest="es_patience", type=int, default=10)
    parser.add_argument("--accum-step", dest="accum_step", type=int, default=16)
    parser.add_argument("--scheduler-step", dest="sc_step", type=int, default=100)
    parser.add_argument("--scheduler-gamma", dest="sc_gamma", type=float, default=0.5)
    parser.add_argument("--warmup", dest="warmup", type=int, default=1)
    parser.add_argument("--extra", dest="extra", type=str, default="")

    parser.add_argument("--seed", dest="seed", type=int, default=42)

    args: Namespace = parser.parse_args()  # type: ignore

    assert args.model in __supported_models__, f"model {args.model} not supported"
    assert args.device in __supported_device__, f"device {args.device} not supported"

    torch.set_float32_matmul_precision("medium")

    if args.debug:
        ############## debug ###############
        import debugpy

        debugpy.listen(("localhost", 5678))
        debugpy.wait_for_client()

    seed_everything(args.seed)

    main(args)
