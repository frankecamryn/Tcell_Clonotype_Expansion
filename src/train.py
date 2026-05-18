'''
Training loops for the MLP and supervised autoencoder!
'''

from __future__ import annotations
import random
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import MLPClassifier, MultiTaskModel, SupervisedAutoencoder


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int = 512, shuffle: bool = True
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

def pos_weight_from(y: np.ndarray) -> torch.Tensor:
    # BCEWithLogitsLoss pos_weight so the positive class isn't drowned out
    pos = y.sum()
    neg = len(y) - pos
    return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    recon_weight: float = 1.0   # SAE only
    clf_weight: float = 1.0     # SAE only
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    history: dict = field(default_factory=dict)

def train_mlp(
    X_train, y_train, X_val, y_val, in_features: int, cfg: TrainConfig | None = None,
    hidden_dims: tuple[int, ...] = (512, 128), dropout: float = 0.3,
) -> tuple[MLPClassifier, dict]:
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    model = MLPClassifier(in_features, hidden_dims=hidden_dims, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_from(y_train).to(device))

    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            logit = model(xb)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch+1:3d}/{cfg.epochs}  train={tr_loss:.4f}  val={val_loss:.4f}")

    return model, history


def train_sae(
    X_train, y_train, X_val, y_val, in_features: int, cfg: TrainConfig | None = None,
    hidden_dims: tuple[int, ...] = (512, 128), latent_dim: int = 32, dropout: float = 0.2,
) -> tuple[SupervisedAutoencoder, dict]:
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    model = SupervisedAutoencoder(
        in_features, hidden_dims=hidden_dims, latent_dim=latent_dim, dropout=dropout
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    recon_fn = nn.MSELoss()
    clf_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_from(y_train).to(device))

    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)

    history = {"train_loss": [], "train_recon": [], "train_clf": [],
               "val_loss": [], "val_recon": [], "val_clf": []}

    for epoch in range(cfg.epochs):
        model.train()
        sums = {"loss": 0.0, "recon": 0.0, "clf": 0.0}
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            recon, logit, _ = model(xb)
            r = recon_fn(recon, xb)
            c = clf_fn(logit, yb)
            loss = cfg.recon_weight * r + cfg.clf_weight * c
            loss.backward()
            opt.step()
            sums["loss"] += loss.item() * xb.size(0)
            sums["recon"] += r.item() * xb.size(0)
            sums["clf"] += c.item() * xb.size(0)
        n = len(train_loader.dataset)
        history["train_loss"].append(sums["loss"] / n)
        history["train_recon"].append(sums["recon"] / n)
        history["train_clf"].append(sums["clf"] / n)

        model.eval()
        vsums = {"loss": 0.0, "recon": 0.0, "clf": 0.0}
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon, logit, _ = model(xb)
                r = recon_fn(recon, xb)
                c = clf_fn(logit, yb)
                loss = cfg.recon_weight * r + cfg.clf_weight * c
                vsums["loss"] += loss.item() * xb.size(0)
                vsums["recon"] += r.item() * xb.size(0)
                vsums["clf"] += c.item() * xb.size(0)
        n = len(val_loader.dataset)
        history["val_loss"].append(vsums["loss"] / n)
        history["val_recon"].append(vsums["recon"] / n)
        history["val_clf"].append(vsums["clf"] / n)
        print(
            f"epoch {epoch+1:3d}/{cfg.epochs}  "
            f"train: loss={history['train_loss'][-1]:.4f} recon={history['train_recon'][-1]:.4f} clf={history['train_clf'][-1]:.4f}  "
            f"val: loss={history['val_loss'][-1]:.4f} recon={history['val_recon'][-1]:.4f} clf={history['val_clf'][-1]:.4f}"
        )

    return model, history


def train_sae_finetune(
    model: SupervisedAutoencoder,
    X_train, y_train, X_val, y_val,
    cfg: TrainConfig | None = None,
    freeze_encoder: bool = True,
) -> tuple[SupervisedAutoencoder, dict]:
    """
    stage 2 SAE training:
    - takes a model that's already been pretrained for reconstruction 
    -fits just the classifier head on top of its latent representation
    - by default the encoder weights are frozen so  latent space is fixed
    (set `freeze_encoder=False` for fine-tuning)
    """
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.encoder.eval()

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters — did you freeze everything?")
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_from(y_train).to(device))

    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.epochs):
        # decoder + classifier in train mode; encoder stays in eval if frozen
        model.decoder.train()
        model.classifier.train()
        if not freeze_encoder:
            model.encoder.train()

        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            _, logit, _ = model(xb)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, logit, _ = model(xb)
                val_loss += loss_fn(logit, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch+1:3d}/{cfg.epochs}  train={tr_loss:.4f}  val={val_loss:.4f}")

    return model, history


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 1024,
                  device: str | None = None) -> np.ndarray:
    # run the model in eval mode and return sigmoid (logit) for the pos class.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size].astype(np.float32)).to(device)
        out = model(xb)
        if isinstance(out, tuple):
            _, logit, _ = out
        else:
            logit = out
        probs.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(probs)


@torch.no_grad()
def encode_latents(model, X: np.ndarray, batch_size: int = 1024,
                   device: str | None = None) -> np.ndarray:
    # project X into the encoder's latent space; works for SAE and MultiTaskModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    zs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size].astype(np.float32)).to(device)
        zs.append(model.encode(xb).cpu().numpy())
    return np.concatenate(zs)


# ---------------------------------------------------------------------------
# Multitask training!
# ---------------------------------------------------------------------------


def make_loader_multi(
    X: np.ndarray,
    y_expanded: np.ndarray,
    y_clone_bin: np.ndarray,
    y_ident: np.ndarray,
    batch_size: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    # dataLoader yielding (x, y_expanded, y_clone_bin, y_ident) tuple
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y_expanded.astype(np.float32)),
        torch.from_numpy(y_clone_bin.astype(np.int64)),
        torch.from_numpy(y_ident.astype(np.int64)),
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True
    )


def class_weights_from(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Inverse frequency class weights for CrossEntropyLoss.
    - values outside [0, n_classes) are treated as missing-label sentinels
        (e.g -1 from pd.Categorical.codes) and excluded from the counts
    """
    y = np.asarray(y).astype(np.int64)
    mask = (y >= 0) & (y < n_classes)
    counts = np.bincount(y[mask], minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


@dataclass
class MultiTaskConfig:
    """
    Hyperparameters for multitask training.
    - The three w_* knobs control how the losses combine:
        loss = w_expanded * BCE(expanded) + w_clone * CE(clone_bin) + w_ident * CE(ident)
    """
    epochs: int = 30
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    w_expanded: float = 1.0
    w_clone: float = 0.5
    w_ident: float = 0.5
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_multitask(
    X_train: np.ndarray,
    y_train_expanded: np.ndarray,
    y_train_clone_bin: np.ndarray,
    y_train_ident: np.ndarray,
    X_val: np.ndarray,
    y_val_expanded: np.ndarray,
    y_val_clone_bin: np.ndarray,
    y_val_ident: np.ndarray,
    n_clone_bins: int,
    n_idents: int,
    cfg: MultiTaskConfig | None = None,
    hidden_dims: tuple[int, ...] = (512, 128),
    latent_dim: int = 64,
    dropout: float = 0.2,
) -> tuple[MultiTaskModel, dict]:
    #train the multitask model; returns the model and per-epoch history
    cfg = cfg or MultiTaskConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    model = MultiTaskModel(
        in_features=X_train.shape[1],
        n_clone_bins=n_clone_bins,
        n_idents=n_idents,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_from(y_train_expanded).to(device))
    # ignore_index=-1 lets cells with missing auxlabels (e.g clone_bin
    # nan from pd.cut) skip those tasks without crashing/ dragging the loss
    ce_clone = nn.CrossEntropyLoss(
        weight=class_weights_from(y_train_clone_bin, n_clone_bins).to(device),
        ignore_index=-1,
    )
    ce_ident = nn.CrossEntropyLoss(
        weight=class_weights_from(y_train_ident, n_idents).to(device),
        ignore_index=-1,
    )

    train_loader = make_loader_multi(
        X_train, y_train_expanded, y_train_clone_bin, y_train_ident,
        batch_size=cfg.batch_size, shuffle=True,
    )
    val_loader = make_loader_multi(
        X_val, y_val_expanded, y_val_clone_bin, y_val_ident,
        batch_size=cfg.batch_size, shuffle=False,
    )

    history = {
        "train_total": [], "train_expanded": [], "train_clone": [], "train_ident": [],
        "val_total": [],   "val_expanded": [],   "val_clone": [],   "val_ident": [],
    }

    def _step(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()
        sums = {"total": 0.0, "expanded": 0.0, "clone": 0.0, "ident": 0.0}
        n = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for xb, y_e, y_c, y_i in loader:
                xb = xb.to(device, non_blocking=True)
                y_e = y_e.to(device, non_blocking=True)
                y_c = y_c.to(device, non_blocking=True)
                y_i = y_i.to(device, non_blocking=True)
                if train:
                    opt.zero_grad()
                out = model(xb)
                loss_e = bce(out["expanded"], y_e)
                loss_c = ce_clone(out["clone_bin"], y_c)
                loss_i = ce_ident(out["ident"], y_i)
                loss = (cfg.w_expanded * loss_e
                        + cfg.w_clone * loss_c
                        + cfg.w_ident * loss_i)
                if train:
                    loss.backward()
                    opt.step()
                bs = xb.size(0)
                sums["total"]    += loss.item()   * bs
                sums["expanded"] += loss_e.item() * bs
                sums["clone"]    += loss_c.item() * bs
                sums["ident"]    += loss_i.item() * bs
                n += bs
        return {k: v / n for k, v in sums.items()}

    for epoch in range(cfg.epochs):
        tr = _step(train_loader, train=True)
        va = _step(val_loader,   train=False)
        for k in ["total", "expanded", "clone", "ident"]:
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(va[k])
        print(
            f"epoch {epoch+1:3d}/{cfg.epochs}  "
            f"train: total={tr['total']:.4f} exp={tr['expanded']:.4f} "
            f"clone={tr['clone']:.4f} ident={tr['ident']:.4f}  "
            f"val: total={va['total']:.4f} exp={va['expanded']:.4f} "
            f"clone={va['clone']:.4f} ident={va['ident']:.4f}"
        )

    return model, history


@torch.no_grad()
def predict_multitask(
    model: MultiTaskModel, X: np.ndarray, batch_size: int = 1024,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Return dict with arrays:
        -p_expanded: sigmoid prob of expanded
        -p_clone_bin: softmax probs (n, n_clone_bins)
        -p_ident: softmax probs (n, n_idents)
        -z: latent embedding (n, latent_dim)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    out = {"p_expanded": [], "p_clone_bin": [], "p_ident": [], "z": []}
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size].astype(np.float32)).to(device)
        o = model(xb)
        out["p_expanded"].append(torch.sigmoid(o["expanded"]).cpu().numpy())
        out["p_clone_bin"].append(torch.softmax(o["clone_bin"], dim=-1).cpu().numpy())
        out["p_ident"].append(torch.softmax(o["ident"], dim=-1).cpu().numpy())
        out["z"].append(o["z"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}
