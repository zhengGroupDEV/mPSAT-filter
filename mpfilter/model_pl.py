import copy
import os
import random
import warnings
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as tm
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .config import (
    ConfigAe,
    ConfigBase,
    ConfigCconv,
    ConfigConv,
)
from .model import (
    MpModelClsConv,
    MpModelSeqAe,
    MpModelSeqConv,
)

matplotlib.use("Agg")


class MpSaveCkptOnShutdown(Callback):
    def __init__(self, saveto: str = "ckpts/interrupt.ckpt") -> None:
        super().__init__()
        self.saveto = saveto

    def on_exception(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        exception: BaseException,
    ) -> None:
        trainer.save_checkpoint(self.saveto)
        print(f"Exception detected, saved to {self.saveto}")


class StepLrWithWarmup(optim.lr_scheduler.StepLR):
    def __init__(
        self,
        optimizer,
        step_size: int,
        warmup_step: int,
        gamma=0.1,
        last_epoch=-1,
        verbose=False,
    ) -> None:
        self.warmup_step = warmup_step
        self.init_lr_groups = copy.deepcopy(optimizer.param_groups)
        super(StepLrWithWarmup, self).__init__(optimizer, step_size, gamma, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:  # type: ignore
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self._step_count <= self.warmup_step:  # type: ignore
            lr_scale = min(1.0, float(self._step_count) / self.warmup_step)  # type: ignore
            lr = [group["lr"] * lr_scale for group in self.init_lr_groups]
            return lr
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):  # type: ignore
            lr = [group["lr"] for group in self.optimizer.param_groups]  # type: ignore
            return lr
        lr = [group["lr"] * self.gamma for group in self.optimizer.param_groups]  # type: ignore
        return lr


class MpModelBaseLight(pl.LightningModule):
    def __init__(
        self,
        optimizer: str = "adam",
        lr: float = 1e-4,
        sc_step: int = 100,
        sc_gamma: float = 0.5,
        warmup_step: int = 1,
    ) -> None:
        super(MpModelBaseLight, self).__init__()
        self.optim = optimizer
        self.lr = lr
        self.sc_step = sc_step
        self.sc_gamma = sc_gamma
        self.warmup_step = warmup_step

    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"optimizer {self.optim} not implemented")
        scheduler = {
            "scheduler": StepLrWithWarmup(
                optimizer,
                step_size=self.sc_step,
                gamma=self.sc_gamma,
                warmup_step=self.warmup_step,
            ),
            "name": "lr_scheduler",
            # "monitor": "train/loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]


###########################################
#              classification             #
###########################################
class MpModelClsLight(MpModelBaseLight):
    def __init__(
        self,
        conf: ConfigCconv,
        model: MpModelClsConv,
        optimizer: str = "adam",
        lr: float = 1e-4,
        sc_step: int = 100,
        sc_gamma: float = 0.5,
        warmup_step: int = 1,
    ) -> None:
        super(MpModelClsLight, self).__init__(
            optimizer=optimizer,
            lr=lr,
            sc_step=sc_step,
            sc_gamma=sc_gamma,
            warmup_step=warmup_step,
        )
        self.model = model
        t = torch.randint(0, 1000, (conf.num_class, 3600)) / conf.num_tokens
        self.example_input_array = (t, torch.arange(conf.num_class))

        # metrics
        self.accuracy_score = tm.Accuracy(
            task="multiclass",
            num_classes=conf.num_class,
            average="macro",
        )
        self.train_accuracy_score = tm.Accuracy(
            task="multiclass",
            num_classes=conf.num_class,
            average="macro",
        )
        self.precision_score = tm.Precision(
            task="multiclass",
            num_classes=conf.num_class,
            average="macro",
        )
        self.recall_score = tm.Recall(
            task="multiclass",
            num_classes=conf.num_class,
            average="macro",
        )
        self.f1_score = tm.F1Score(
            task="multiclass",
            num_classes=conf.num_class,
            average="macro",
        )
        # self.cf_matrix = tm.ConfusionMatrix(
        #     num_classes=conf.num_class,
        # )

        # plot data
        self.plot_cfm_preds = [i for i in range(conf.num_class)]
        self.plot_cfm_trues = [i for i in range(conf.num_class)]
        self.test_trues = []
        self.test_preds = []

    def forward(self, src, label):
        logits = self.model(src.unsqueeze(1))
        return logits

    def training_step(self, batch, batch_idx):
        src, labels = batch
        logits = self.model(src.unsqueeze(1))
        loss = F.cross_entropy(logits, labels)
        probs = F.softmax(logits, dim=-1)
        self.train_accuracy_score(probs, labels)
        self.log("train/loss", loss)
        self.log("train/accuracy", self.train_accuracy_score)
        return loss

    def validation_step(self, batch, batch_idx):
        src, labels = batch
        logits = self.model(src.unsqueeze(1))
        loss = F.cross_entropy(logits, labels)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        self.accuracy_score(probs, labels)
        self.precision_score(preds, labels)
        self.recall_score(preds, labels)

        self.log("val/loss", loss)
        self.log("val/accuracy", self.accuracy_score)
        self.log("val/precision", self.precision_score)
        self.log("val/recall", self.recall_score)

        self.plot_cfm_preds += preds.tolist()
        self.plot_cfm_trues += labels.tolist()
        return loss

    def on_validation_end(self) -> None:
        self.plot_cf_matrix(
            "val/plot_cfm",
            self.plot_cfm_preds,
            self.plot_cfm_trues,
        )
        self.plot_cfm_preds.clear()
        self.plot_cfm_trues.clear()

    def test_step(self, batch, batch_idx):
        src, labels = batch
        logits = self.model(src.unsqueeze(1))
        loss = F.cross_entropy(logits, labels)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        # self.test_trues.extend(labels.tolist())
        # self.test_preds.extend(preds.tolist())
        self.accuracy_score(probs, labels)
        self.precision_score(preds, labels)
        self.recall_score(preds, labels)
        self.f1_score(preds, labels)
        self.plot_cfm_preds += preds.tolist()
        self.plot_cfm_trues += labels.tolist()

        self.log("test/loss", loss)
        self.log("test/accuracy", self.accuracy_score)
        self.log("test/precision", self.precision_score)
        self.log("test/recall", self.recall_score)
        self.log("test/f1", self.f1_score)
        return loss

    def on_test_end(self) -> None:
        self.plot_cf_matrix(
            "test/plot_cfm",
            self.plot_cfm_preds,
            self.plot_cfm_trues,
        )
        # from sklearn import metrics
        # report = metrics.classification_report(self.test_trues, self.test_preds)
        # report["trues"] = self.test_trues
        # report["preds"] = self.test_preds

    def plot_cf_matrix(self, tag: str, y_pred, y_label):
        # matplotlib.rcParams["font.size"] = 8
        labels = [
            "HDPE",
            "LDPE",
            "LLDPE",
            "PET",
            "ABS",
            "PP",
            "PS",
            "PVC",
            "CA",
            "PMMA",
            "PA",
            "PC",
            "PLA",
            "PBT",
            "CPE",
            "EVA",
            "PU",
            "PTFE",
            "POM",
            "PVDF",
            "PCL",
            "EVOH",
            "PVOH",
            "PVA",
            "UNKNOWN",
        ]
        if max(len(y_label), len(y_pred)) < len(labels) - 1:
            labels = None
        cf_matrix = confusion_matrix(y_label, y_pred)
        # cf_matrix = m.to("cpu").numpy()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        disp = ConfusionMatrixDisplay(cf_matrix, display_labels=labels)
        disp.plot(
            ax=ax,
            cmap="Blues",
            values_format="",
            xticks_rotation="vertical",
            colorbar=False,
            text_kw={"fontsize": 6},
            # include_values=False,
        )
        ax.set_xlabel("Predicted Label", fontdict={"size": 10})
        ax.set_ylabel("Real Label", fontdict={"size": 10})
        fig.colorbar(disp.im_, shrink=0.78, pad=0.01)
        fig.tight_layout()

        tb = self.logger.experiment  # type: ignore
        tb.add_figure(tag, fig, self.trainer.global_step, close=True)


class MpModelCconvLight(MpModelClsLight):
    def __init__(
        self,
        conf: ConfigCconv,
        optimizer: str = "adam",
        lr: float = 0.0001,
        sc_step: int = 100,
        sc_gamma: float = 0.5,
        warmup_step: int = 1,
    ) -> None:
        model = MpModelClsConv(
            in_features=conf.in_features,
            out_features=conf.num_class,
            dropout=conf.dropout,
        )
        super().__init__(
            conf=conf,
            model=model,
            optimizer=optimizer,
            lr=lr,
            sc_step=sc_step,
            sc_gamma=sc_gamma,
            warmup_step=warmup_step,
        )


###########################################
#              reconstruction             #
###########################################
class MpModelSeqLight(MpModelBaseLight):
    def __init__(
        self,
        conf: ConfigBase,
        model: Union[MpModelSeqAe, MpModelSeqConv],
        optimizer: str = "adam",
        lr: float = 1e-4,
        sc_step: int = 100,
        sc_gamma: float = 0.5,
        warmup_step: int = 1,
    ):
        super(MpModelSeqLight, self).__init__(
            optimizer=optimizer,
            lr=lr,
            sc_step=sc_step,
            sc_gamma=sc_gamma,
            warmup_step=warmup_step,
        )
        self.model = model
        self.model_name = conf.model

        t = torch.randint(0, 1000, (4, 3600))

        self.example_input_array = (t, t)
        self.plot_batch: Tuple[torch.Tensor, torch.Tensor] = None  # type: ignore

        # metrics
        self.snr_score = tm.audio.SignalNoiseRatio()  # type: ignore
        self.mse_score = tm.MeanSquaredError()
        self.pearson_score = tm.PearsonCorrCoef()
        self.r2_score = tm.R2Score()

    @property
    def model_(self):
        return self.model

    @model_.setter
    def model_(self, m):
        self.model = m

    def forward(self, src, tgt):
        src = src.unsqueeze(1).to(torch.float32)
        logits = self.model(src)
        return logits

    def training_step(self, batch, batch_idx):
        src = batch[0].unsqueeze(1).to(torch.float32)
        tgt = batch[1].to(torch.float32)
        logits = self.model(src)
        loss = F.mse_loss(logits, tgt)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src = batch[0].unsqueeze(1).to(torch.float32)
        tgt = batch[1].to(torch.float32)
        logits: torch.Tensor = self.model(src)
        loss = F.mse_loss(logits, tgt)

        self.snr_score(logits.flatten(), tgt.flatten())
        self.mse_score(logits.flatten(), tgt.flatten())
        self.pearson_score(logits.flatten(), tgt.flatten())
        self.r2_score(logits.flatten(), tgt.flatten())

        self.log("val/loss", loss)
        self.log("val/snr", self.snr_score)
        self.log("val/mse", self.mse_score)
        self.log("val/pearson", self.pearson_score)
        self.log("val/r2", self.r2_score)

        if self.plot_batch is None:
            self.plot_batch = batch
        return loss

    def on_validation_end(self) -> None:
        self.eval_plot_metric(self.plot_batch)

    def test_step(self, batch, batch_idx):
        src = batch[0].unsqueeze(1).to(torch.float32)
        tgt = batch[1].to(torch.float32)
        logits = self.model(src)

        self.snr_score(logits.flatten(), tgt.flatten())
        self.mse_score(logits.flatten(), tgt.flatten())
        self.pearson_score(logits.flatten(), tgt.flatten())
        self.r2_score(logits.flatten(), tgt.flatten())

        self.log("test/snr", self.snr_score)
        self.log("test/mse", self.mse_score)
        self.log("test/pearson", self.pearson_score)
        self.log("test/r2", self.r2_score)

    def on_test_end(self) -> None:
        ...
        # logits = torch.concat(self.test_pred, dim=0)
        # tgt = torch.concat(self.test_true, dim=0)

    def eval_plot_metric(self, ys: Tuple[torch.Tensor, torch.Tensor]):
        epoch = self.trainer.global_step
        y0, y1 = ys
        tb = self.logger.experiment  # type: ignore

        y2 = self.model(y0.unsqueeze(1))

        fig = plt.figure(figsize=(15, 10), dpi=120)
        n_fig = min(y0.size(0), 5)
        for b in range(n_fig):
            ax = fig.add_subplot(n_fig, 1, b + 1)
            ax.plot(y0[b].cpu().numpy(), label="Input")
            ax.plot(y1[b].cpu().numpy(), label="True")
            ax.plot(y2[b].cpu().numpy(), label="Output")
            ax.legend()
            ax.grid()
        tb.add_figure("val/plot", fig, epoch, close=True)


class MpModelConvLight(MpModelSeqLight):
    def __init__(
        self,
        conf: ConfigConv,
        optimizer: str = "adam",
        lr: float = 0.0001,
        sc_step: int = 100,
        sc_gamma: float = 0.5,
        warmup_step: int = 1,
    ):
        model = MpModelSeqConv(
            in_features=conf.in_features,
            out_features=conf.out_features,
            dropout=conf.dropout,
        )
        super().__init__(conf, model, optimizer, lr, sc_step, sc_gamma, warmup_step)


class MpModelAeLight(MpModelSeqLight):
    def __init__(
        self,
        conf: ConfigAe,
        optimizer: str = "adam",
        lr: float = 0.0001,
        sc_step: int = 100,
        sc_gamma: float = 0.5,
        warmup_step: int = 1,
    ):
        model = MpModelSeqAe(
            in_features=conf.in_features,
            out_features=conf.out_features,
            hid_features=conf.hid_features,
            dropout=conf.dropout,
        )
        super().__init__(conf, model, optimizer, lr, sc_step, sc_gamma, warmup_step)
