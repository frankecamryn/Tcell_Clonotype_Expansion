"""
Model definitions: 
    - MLP classifier
    - Supervised autoencoder
"""
from __future__ import annotations
import torch
from torch import nn

class MLPClassifier(nn.Module):
    # basic feed-forward classifier: input -> hidden layers -> sigmoid logit
    def __init__(
        self,
        in_features: int,
        hidden_dims: tuple[int, ...] = (512, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SupervisedAutoencoder(nn.Module):
    # Encoder + decoder + classifier head sharing the same latent space
    def __init__(
        self,
        in_features: int,
        hidden_dims: tuple[int, ...] = (512, 128),
        latent_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        enc: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        dec.append(nn.Linear(prev, in_features))
        self.decoder = nn.Sequential(*dec)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        logit = self.classifier(z).squeeze(-1)
        return recon, logit, z


class MultiTaskModel(nn.Module):
    """
    Shared encoder that feeds three classification heads
    tasks:
        -expanded: binary (BCEWithLogitsLoss)
        -clone_bin:multi-class ordinal (CrossEntropyLoss)
        -ident: multi-class (T cell state, CrossEntropyLoss)
    """

    def __init__(
        self,
        in_features: int,
        n_clone_bins: int,
        n_idents: int,
        hidden_dims: tuple[int, ...] = (512, 128),
        latent_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        enc: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        def head(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, out_dim),
            )

        self.head_expanded = head(1)
        self.head_clone_bin = head(n_clone_bins)
        self.head_ident = head(n_idents)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {
            "expanded": self.head_expanded(z).squeeze(-1),
            "clone_bin": self.head_clone_bin(z),
            "ident": self.head_ident(z),
            "z": z,
        }
