import torch
import torch.nn as nn

from .config import ConfigAE, ConfigVCNN
from .model import MpAE, MpVCNN


class InferNN(nn.Module):
    def __init__(
        self, model_path: str, conf_path: str, model_name: str = "ae", device="cuda"
    ) -> None:
        super().__init__()
        if model_name == "ae":
            self.conf = ConfigAE(conf_path, init=False)
            self.filter = MpAE(
                in_features=self.conf.in_features,
                out_features=self.conf.out_features,
                hid_features=self.conf.hid_features,
                dropout=self.conf.dropout,
            )
        elif model_name == "vcnn":
            self.conf = ConfigVCNN(conf_path, init=False)
            self.filter = MpVCNN(
                in_features=self.conf.in_features,
                out_features=self.conf.out_features,
                dropout=self.conf.dropout,
            )
        else:
            raise ValueError(f"model name is {model_name}")

        self.filter.load_state_dict(torch.load(model_path, map_location=device))

        self.filter.eval()

    def forward(self, x):
        # x: (3600,)
        return self.filter(x)
