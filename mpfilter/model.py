from typing import List, Optional, Tuple

import einops
import torch
import torch.nn as nn


#######################################
######## Reconstruction  #############
######################################
class MpAE(nn.Module):
    def __init__(
        self,
        in_features: int = 3600,
        out_features: int = 3600,
        hid_features: int = 128,
        dropout=0.2,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2048),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            self.dropout,
            nn.Linear(in_features=1024, out_features=512),
            nn.Linear(in_features=512, out_features=256),
            nn.Linear(in_features=256, out_features=hid_features),
            nn.ReLU(),
            self.dropout,
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=hid_features, out_features=256),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(in_features=512, out_features=1024),
            nn.Linear(in_features=1024, out_features=2048),
            nn.ReLU(),
            self.dropout,
            nn.Linear(in_features=2048, out_features=out_features),
        )

        self.__init_weights__()

    def __init_weights__(self):
        ...

    def forward(self, x: torch.Tensor):
        # x: (B, 1, N), competible with MpVCNN
        x = self.encoder(x.squeeze(1))
        # (B, H)
        x = self.decoder(x)
        # (B, OUT)
        return x


class MpVCNN(nn.Module):
    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 3600,
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
    ) -> None:
        super(MpVCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_features, 4, 11, stride=7),
            nn.BatchNorm1d(4, eps=norm_eps),
            nn.ReLU(),
            nn.Conv1d(4, 64, 9, stride=5),
            nn.BatchNorm1d(64, eps=norm_eps),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Conv1d(64, 256, 7, stride=3),
            nn.BatchNorm1d(256, eps=norm_eps),
            nn.ReLU(),
            nn.Conv1d(256, 512, 5, stride=1),
            nn.BatchNorm1d(512, eps=norm_eps),
            nn.ReLU(),
            nn.Conv1d(512, 768, 3, stride=1),
            nn.BatchNorm1d(768, eps=norm_eps),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.ConvTranspose1d(768, 512, 3, stride=1),
            nn.BatchNorm1d(512, eps=norm_eps),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, 5, stride=1),
            nn.BatchNorm1d(256, eps=norm_eps),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 64, 7, stride=3),
            nn.BatchNorm1d(64, eps=norm_eps),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 4, 9, stride=5),
            nn.BatchNorm1d(4, eps=norm_eps),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 1, 11, stride=9),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(nn.Linear(1 * 4538, out_features))

    def forward(self, x):
        x = self.encoder(x)
        x = einops.rearrange(x, "b d l -> b (d l)")
        x = self.decoder(x)
        return x


#######################################
######## Classification  #############
######################################
class MpCNN1D(nn.Module):
    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 25,
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
    ) -> None:
        super(MpCNN1D, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_features, 4, 11, stride=7),
            nn.BatchNorm1d(4, eps=norm_eps),
            nn.GELU(),
            nn.Conv1d(4, 16, 7, stride=5),
            nn.BatchNorm1d(16, eps=norm_eps),
            nn.Conv1d(16, 64, 5, stride=3),
            nn.BatchNorm1d(64, eps=norm_eps),
            nn.Dropout(p=dropout),
            nn.Conv1d(64, 256, 3, stride=3),
            nn.BatchNorm1d(256, eps=norm_eps),
            nn.Conv1d(256, 1024, 3, stride=3),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024 * 3, 1024),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, out_features),
        )

    def forward(self, x):
        x = self.encoder(x.to(torch.float32))
        x = einops.rearrange(x, "b d l -> b (d l)")
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    device = "cpu"
    ...
