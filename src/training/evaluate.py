"""One final test evaluation per frozen split, with metrics, confusion matrices and actual errors."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader

from src.data.make_dataset import load_partition
from src.data.splits import ATTRIBUTES, file_hash
from src.models.architecture import CustomMultiHeadCNN
from src.training.loops import predictions


def attribute_metrics(trues, preds):
    metrics = {}
    for attr, (_, classes) in ATTRIBUTES.items():
        labels = list(range(classes))
        true, pred = np.asarray(trues[attr]), np.asarray(preds[attr])
        if len(true) == 0 or len(true) != len(pred):
            raise ValueError("Predictions and labels must be aligned and nonempty")
        if not np.isin(true, labels).all() or not np.isin(pred, labels).all():
            raise ValueError(f"Invalid class index: {attr}")
        report = classification_report(true, pred, labels=labels, output_dict=True, zero_division=0)
        metrics[attr] = {"accuracy": float(accuracy_score(true, pred)),
                         "macro_f1": float(report["macro avg"]["f1-score"]),
                         "weighted_f1": float(report["weighted avg"]["f1-score"]),
                         "support": len(true), "labels": labels,
                         "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(),
                         "classification_report": report}
    return metrics


def publish_report(metrics, provenance, records, dataset, output_dir, max_errors=8):
    output_dir = Path(output_dir)
    pd.DataFrame(records).to_csv(output_dir / "predictions.csv", index=False)
    errors = []
    lines = ["# Reserved test evaluation", "", "Results computed from model predictions on the frozen test manifest.",
             "Class indices follow the annotation CSV. Macro F1 includes every defined class (absent class = 0).", "",
             "| Attribute | Accuracy | Macro F1 | Weighted F1 | Support |",
             "|---|---:|---:|---:|---:|"]
    for attr, values in metrics.items():
        lines.append(f"| {attr} | {values['accuracy']:.4f} | {values['macro_f1']:.4f} | "
                     f"{values['weighted_f1']:.4f} | {values['support']} |")
        cm = np.asarray(values["confusion_matrix"])
        pd.DataFrame(cm, index=values["labels"], columns=values["labels"]).to_csv(output_dir / f"{attr}_confusion.csv")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set(xlabel="Predicted", ylabel="True", title=attr,
               xticks=values["labels"], yticks=values["labels"])
        for (i, j), count in np.ndenumerate(cm):
            ax.text(j, i, str(count), ha="center", va="center", color="white" if count > cm.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(output_dir / f"{attr}_confusion.png")
        plt.close(fig)
        failures = [(i, r) for i, r in enumerate(records) if r[f"true_{attr}"] != r[f"pred_{attr}"]]
        failures.sort(key=lambda pair: (-pair[1][f"confidence_{attr}"], pair[1]["filename"]))
        chosen = failures[:max_errors]
        for i, row in chosen:
            errors.append({"attribute": attr, "filename": row["filename"], "true": row[f"true_{attr}"],
                           "predicted": row[f"pred_{attr}"], "confidence": row[f"confidence_{attr}"]})
        if chosen:
            fig, axes = plt.subplots(1, len(chosen), figsize=(2.6 * len(chosen), 3), squeeze=False)
            for ax, (i, row) in zip(axes[0], chosen):
                image, _ = dataset[i]
                ax.imshow(image.permute(1, 2, 0).numpy()[..., ::-1].clip(0, 1))
                ax.set_title(f"{row['filename']}\ntrue={row[f'true_{attr}']} pred={row[f'pred_{attr}']}", fontsize=8)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(output_dir / f"{attr}_errors.png")
            plt.close(fig)
    pd.DataFrame(errors, columns=["attribute", "filename", "true", "predicted", "confidence"]).to_csv(
        output_dir / "errors.csv", index=False)
    lines += ["", "## Protocol and provenance", "", "```json", json.dumps(provenance, indent=2), "```", "",
              "## Confusion matrices and errors", "",
              "Matrices: rows = true labels, columns = predictions. Error examples are the most confident incorrect "
              "predictions per attribute, chosen after the final evaluation; do not use them to retune this test.", ""]
    for attr in metrics:
        lines += [f"### {attr}", "", f"![{attr} confusion matrix]({attr}_confusion.png)", ""]
        if (output_dir / f"{attr}_errors.png").exists():
            lines += [f"![{attr} errors]({attr}_errors.png)", ""]
        else:
            lines += ["No incorrect predictions for this attribute on this test partition.", ""]
    if not provenance["identity_separated"]:
        lines += ["Identity labels were unavailable: this result does not establish identity-disjoint generalisation.", ""]
    lines += ["Near-duplicate hashing is a heuristic, not a guarantee of identity separation.", ""]
    (output_dir / "README.md").write_text("\n".join(lines))


def evaluate_model(*, processed_dir="data/processed", splits_dir="data/splits", model_path="models/best_model.pth",
                   output_dir="reports/test", ledger_dir="data/test_access", batch_size=32, max_errors=8):
    if batch_size < 1 or max_errors < 1:
        raise ValueError("batch_size and max_errors must be positive")
    if Path(output_dir).exists():
        raise FileExistsError("Report directory already exists; final results must not be overwritten")
    dataset, bundle, metadata = load_partition("test", processed_dir, splits_dir)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("protocol_version") != 1 or checkpoint.get("split_id") != metadata["split_id"] or
            checkpoint.get("selection_partition") != "validation" or checkpoint.get("selection_metric") != "validation_loss"):
        raise ValueError("Checkpoint must be selected on validation under this exact frozen split; legacy models are rejected")
    # Verify the train/validation data used for selection before consuming the reserved test.
    for name in ("train", "validation"):
        _, selected_bundle, _ = load_partition(name, processed_dir, splits_dir)
        if checkpoint[f"{name}_tensors_sha256"] != selected_bundle["tensors_sha256"]:
            raise ValueError("Training/validation tensor provenance mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomMultiHeadCNN(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    ledger_dir = Path(ledger_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger = ledger_dir / f"{metadata['split_id']}.json"
    provenance = {"protocol_version": 1, "partition": "test", "split_id": metadata["split_id"],
                  "checkpoint_sha256": file_hash(model_path), "selected_epoch": checkpoint["epoch"],
                  "selection_metric": "validation_loss", "test_tensors_sha256": bundle["tensors_sha256"],
                  "identity_separated": metadata["audit"]["identity_separated"],
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "started"}
    # Exclusive creation applies to the split, not just to one checkpoint/output path.
    try:
        with ledger.open("x") as stream:
            json.dump(provenance, stream, indent=2)
    except FileExistsError as exc:
        raise RuntimeError("Reserved test already accessed for this split; evaluation is not repeatable for model selection") from exc
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=False)
        trues, preds, records = {k: [] for k in ATTRIBUTES}, {k: [] for k in ATTRIBUTES}, []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.inference_mode():
            for images, targets in loader:
                outputs = model(images.to(device))
                if not all(torch.isfinite(value).all() for value in outputs.values()):
                    raise ValueError("Non-finite test predictions")
                predicted = predictions(outputs)
                offset = len(records)
                rows = [{"filename": name} for name in bundle["filenames"][offset:offset + len(images)]]
                for attr, (_, classes) in ATTRIBUTES.items():
                    y, p = targets[attr].long().tolist(), predicted[attr].cpu().tolist()
                    trues[attr].extend(y)
                    preds[attr].extend(p)
                    if classes == 2:
                        positive = torch.sigmoid(outputs[attr]).cpu()
                        confidence = torch.maximum(positive, 1 - positive).tolist()
                    else:
                        confidence = outputs[attr].softmax(1).max(1).values.cpu().tolist()
                    for row, true, pred, conf in zip(rows, y, p, confidence):
                        row.update({f"true_{attr}": true, f"pred_{attr}": pred, f"confidence_{attr}": conf})
                records.extend(rows)
        metrics = attribute_metrics(trues, preds)
        provenance["status"] = "completed"
        publish_report(metrics, provenance, records, dataset, output_dir, max_errors)
        result = {"provenance": provenance, "attributes": metrics}
        (Path(output_dir) / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        ledger.write_text(json.dumps(provenance, indent=2) + "\n")
        return result
    except Exception as exc:
        provenance.update(status="failed_after_test_access", error=str(exc))
        ledger.write_text(json.dumps(provenance, indent=2) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--model-path", default="models/best_model.pth")
    parser.add_argument("--output-dir", default="reports/test")
    parser.add_argument("--ledger-dir", default="data/test_access")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-errors", type=int, default=8)
    evaluate_model(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
