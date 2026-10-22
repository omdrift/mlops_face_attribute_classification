"""Tune exclusively on the persisted train and validation partitions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.make_dataset import load_partition
from src.models.architecture import CustomMultiHeadCNN
from src.training.loops import train_one_epoch, validate
from src.training.train import seed_everything


def run_hyperopt(max_evals=20, epochs=5, processed_dir="data/processed", splits_dir="data/splits",
                 output="src/training/hyperopt_params.json", seed=42):
    from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
    if max_evals < 1 or epochs < 1:
        raise ValueError("max_evals and epochs must be positive")
    train_data, _, metadata = load_partition("train", processed_dir, splits_dir)
    val_data, _, _ = load_partition("validation", processed_dir, splits_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    space = {"lr": hp.loguniform("lr", np.log(1e-5), np.log(5e-3)),
             "batch_size": hp.choice("batch_size", [32, 64]),
             "weight_decay": hp.loguniform("weight_decay", np.log(1e-4), np.log(5e-3)),
             "dropout": hp.uniform("dropout", 0.1, 0.6)}

    def objective(params):
        seed_everything(seed)
        train_loader = DataLoader(train_data, batch_size=int(params["batch_size"]), shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_data, batch_size=int(params["batch_size"]))
        model = CustomMultiHeadCNN(n_color=5, n_length=3, dropout=float(params["dropout"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]),
                                      weight_decay=float(params["weight_decay"]))
        best_loss = float("inf")
        for epoch in range(1, epochs + 1):
            train_one_epoch(model, train_loader, optimizer, device, epoch)
            loss, _ = validate(model, val_loader, device)
            best_loss = min(best_loss, loss)
        return {"loss": best_loss, "status": STATUS_OK}

    trials = Trials()
    encoded = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials,
                   rstate=np.random.default_rng(seed))
    selected = space_eval(space, encoded)
    selected = {k: int(v) if k == "batch_size" else float(v) for k, v in selected.items()}
    result = {"split_id": metadata["split_id"], "selection_partition": "validation",
              "selection_metric": "validation_loss", "max_evals": max_evals, "epochs_per_trial": epochs,
              "seed": seed, "params": selected, "best_val_loss": float(trials.best_trial["result"]["loss"])}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-evals", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--output", default="src/training/hyperopt_params.json")
    parser.add_argument("--seed", type=int, default=42)
    run_hyperopt(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
