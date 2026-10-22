"""Select a checkpoint only on frozen validation data; never open test tensors."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.make_dataset import load_partition
from src.models.architecture import CustomMultiHeadCNN
from src.training.loops import train_one_epoch, validate


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_training(*, processed_dir="data/processed", splits_dir="data/splits", model_path="models/best_model.pth",
                 metrics_path="metrics/train_metrics.json", epochs=10, batch_size=32, lr=0.001,
                 weight_decay=0.0001, dropout=0.3, patience=7, seed=42, workers=0,
                 hyperopt_params=None, use_mlflow=False):
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    seed_everything(seed)
    train_data, train_bundle, metadata = load_partition("train", processed_dir, splits_dir)
    val_data, val_bundle, _ = load_partition("validation", processed_dir, splits_dir)
    if hyperopt_params:
        tuned = json.loads(Path(hyperopt_params).read_text())
        if tuned.get("split_id") != metadata["split_id"]:
            raise ValueError("Hyperparameters were selected on a different split")
        selected = tuned["params"]
        batch_size, lr = int(selected["batch_size"]), float(selected["lr"])
        weight_decay, dropout = float(selected["weight_decay"]), float(selected["dropout"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {"n_color": 5, "n_length": 3, "dropout": dropout}
    model = CustomMultiHeadCNN(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    config = {"epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
              "dropout": dropout, "seed": seed, "patience": patience, "selection_metric": "validation_loss"}
    tracking = None
    if use_mlflow:
        import mlflow
        tracking = mlflow
        tracking.set_tracking_uri("sqlite:///mlflow.db")
        tracking.set_experiment("FaceAttributes_FrozenSplits")
    best_loss, best_epoch, waiting, best_accs, history = math.inf, 0, 0, {}, []
    model_path, metrics_path = Path(model_path), Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with tracking.start_run() if tracking else nullcontext():
        if tracking:
            tracking.log_params({**config, "split_id": metadata["split_id"]})
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
            val_loss, val_accs = validate(model, val_loader, device)
            if not math.isfinite(val_loss):
                raise ValueError("Non-finite validation loss")
            scheduler.step(val_loss)
            history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                            "validation_accuracy": val_accs})
            if tracking:
                tracking.log_metrics({"train_loss": train_loss, "validation_loss": val_loss,
                                      **{f"val_acc_{k}": v for k, v in val_accs.items()}}, step=epoch)
            if val_loss < best_loss:
                best_loss, best_epoch, best_accs, waiting = val_loss, epoch, val_accs.copy(), 0
                checkpoint = {"protocol_version": 1, "split_id": metadata["split_id"],
                              "selection_partition": "validation", "selection_metric": "validation_loss",
                              "epoch": epoch, "val_loss": val_loss, "val_accs": best_accs,
                              "model_kwargs": model_kwargs, "model_state_dict": model.state_dict(),
                              "train_tensors_sha256": train_bundle["tensors_sha256"],
                              "validation_tensors_sha256": val_bundle["tensors_sha256"], "config": config}
                temporary = model_path.with_suffix(".tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(model_path)
            else:
                waiting += 1
            print(f"Epoch {epoch}: train={train_loss:.4f} validation={val_loss:.4f}; best epoch={best_epoch}")
            if waiting >= patience:
                break
        result = {"protocol_version": 1, "split_id": metadata["split_id"], "partition": "validation",
                  "selection_metric": "validation_loss", "best_val_loss": best_loss, "best_epoch": best_epoch,
                  "final_epoch": epoch, "best_accuracies": best_accs,
                  "avg_best_accuracy": sum(best_accs.values()) / len(best_accs),
                  "early_stopped": waiting >= patience, "config": config, "history": history}
        metrics_path.write_text(json.dumps(result, indent=2) + "\n")
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ("train_loss", "validation_loss"):
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
        ax.set(xlabel="Epoch", ylabel="Loss", title="Frozen train and validation partitions")
        ax.legend()
        fig.tight_layout()
        plot_path = metrics_path.parent / "training_curves.png"
        fig.savefig(plot_path)
        plt.close(fig)
        if tracking:
            tracking.log_artifact(str(metrics_path))
            tracking.log_artifact(str(plot_path))
            tracking.log_artifact(str(model_path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--model-path", default="models/best_model.pth")
    parser.add_argument("--metrics-path", default="metrics/train_metrics.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--hyperopt-params")
    parser.add_argument("--mlflow", action="store_true", dest="use_mlflow")
    run_training(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
