"""HW2a: convolutional autoencoder anomaly detection on Fashion-MNIST.

Run:
    python autoencoder_invoice_anomaly.py --epochs 5 --max-normal 12000

The script trains only on Fashion-MNIST classes 0-5, selects a threshold from
normal validation errors, and evaluates classes 6-9 as suspicious examples.
It also trains a small dense baseline and writes metrics, plots, examples,
and checkpoints to --out.
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConvAE(nn.Module):
    def __init__(self, bottleneck_channels=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, bottleneck_channels, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channels, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )
    def forward(self, x): return self.decoder(self.encoder(x))
    def encode(self, x): return self.encoder(x)


class DenseAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x)).view(-1, 1, 28, 28)
    def encode(self, x): return self.encoder(x)


def train(model, loader, device, epochs, out, name):
    model.to(device); opt = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.MSELoss()
    history, states = [], []
    for epoch in range(epochs):
        model.train(); total = 0.0
        for x, _ in loader:
            x = x.to(device); opt.zero_grad(); loss = loss_fn(model(x), x); loss.backward(); opt.step()
            total += loss.item() * len(x)
        epoch_loss = total / len(loader.dataset); history.append(epoch_loss)
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"{name} epoch {epoch+1}/{epochs}: {epoch_loss:.6f}")
    torch.save(states, out / f"{name}_states.pt")
    return history


@torch.no_grad()
def errors(model, loader, device):
    model.eval(); vals, labels, images, recon = [], [], [], []
    for x, y in loader:
        pred = model(x.to(device)).cpu(); vals.extend(((x - pred) ** 2).flatten(1).mean(1).numpy())
        labels.extend(y.numpy()); images.extend(x); recon.extend(pred)
    return np.asarray(vals), np.asarray(labels), torch.stack(images), torch.stack(recon)


def plot_examples(images, recon, errs, labels, path, title):
    n = min(6, len(images)); fig, ax = plt.subplots(3, n, figsize=(2*n, 6))
    if n == 1: ax = ax.reshape(3, 1)
    for i in range(n):
        ax[0, i].imshow(images[i, 0], cmap="gray"); ax[0, i].set_title(f"class {labels[i]}");
        ax[1, i].imshow(recon[i, 0], cmap="gray"); ax[1, i].set_title(f"MSE {errs[i]:.4f}");
        ax[2, i].imshow((images[i, 0]-recon[i, 0]).abs(), cmap="magma");
        for r in range(3): ax[r, i].axis("off")
    ax[0, 0].set_ylabel("input"); ax[1, 0].set_ylabel("reconstruction"); ax[2, 0].set_ylabel("absolute error")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def run(args):
    seed_everything(args.seed); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = datasets.FashionMNIST(args.data, train=True, download=True, transform=transforms.ToTensor())
    targets = np.asarray(ds.targets)
    normal_idx = np.flatnonzero(np.isin(targets, np.arange(6)))
    anomaly_idx = np.flatnonzero(np.isin(targets, np.arange(6, 10)))
    rng = np.random.default_rng(args.seed); rng.shuffle(normal_idx); rng.shuffle(anomaly_idx)
    normal_idx = normal_idx[:args.max_normal]; anomaly_idx = anomaly_idx[:args.max_anomaly]
    n_val = min(args.normal_val, len(normal_idx)//4)
    train_idx, val_idx = normal_idx[n_val:], normal_idx[:n_val]
    eval_idx = np.concatenate([val_idx[:min(1000, len(val_idx))], anomaly_idx[:min(500, len(anomaly_idx))]])
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(Subset(ds, eval_idx), batch_size=args.batch_size, shuffle=False)
    normal_eval_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False)

    results = {"config": vars(args) | {"device": str(device), "train_count": len(train_idx), "normal_val_count": len(val_idx), "anomaly_eval_count": min(500, len(anomaly_idx))}}
    for name, model in [("conv", ConvAE(args.bottleneck)), ("dense", DenseAE(args.latent_dim))]:
        history = train(model, train_loader, device, args.epochs, out, name)
        normal_errors, _, _, _ = errors(model, normal_eval_loader, device)
        eval_errors, eval_labels, imgs, rec = errors(model, eval_loader, device)
        threshold = float(np.percentile(normal_errors, args.percentile))
        binary = (eval_labels >= 6).astype(int); pred = (eval_errors > threshold).astype(int)
        cm = confusion_matrix(binary, pred, labels=[0, 1])
        results[name] = {"history": history, "threshold": threshold,
            "precision": float(precision_score(binary, pred, zero_division=0)),
            "recall": float(recall_score(binary, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(binary, eval_errors)), "confusion_matrix": cm.tolist(),
            "normal_error_mean": float(normal_errors.mean()), "anomaly_error_mean": float(eval_errors[binary==1].mean())}
        plt.figure(figsize=(6,4)); plt.plot(history, marker="o"); plt.xlabel("epoch"); plt.ylabel("training MSE"); plt.title(f"{name} training loss"); plt.tight_layout(); plt.savefig(out/f"{name}_loss.png", dpi=160); plt.close()
        plot_examples(imgs[:6], rec[:6], eval_errors[:6], eval_labels[:6], out/f"{name}_normal_examples.png", f"{name} normal examples")
        plot_examples(imgs[-6:], rec[-6:], eval_errors[-6:], eval_labels[-6:], out/f"{name}_anomaly_examples.png", f"{name} suspicious examples")
        plt.figure(figsize=(6,4)); plt.hist(normal_errors, bins=40, alpha=.7, label="normal"); plt.hist(eval_errors[binary==1], bins=40, alpha=.7, label="suspicious"); plt.axvline(threshold, color="black", linestyle="--", label=f"{args.percentile}th percentile"); plt.xlabel("reconstruction MSE"); plt.ylabel("count"); plt.legend(); plt.tight_layout(); plt.savefig(out/f"{name}_error_distribution.png", dpi=160); plt.close()
    (out/"metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", default="data"); p.add_argument("--out", default="outputs"); p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch-size", type=int, default=128); p.add_argument("--max-normal", type=int, default=12000); p.add_argument("--max-anomaly", type=int, default=4000); p.add_argument("--normal-val", type=int, default=1500); p.add_argument("--bottleneck", type=int, default=8); p.add_argument("--latent-dim", type=int, default=32); p.add_argument("--percentile", type=float, default=95); p.add_argument("--seed", type=int, default=42); p.add_argument("--cpu", action="store_true"); run(p.parse_args())
