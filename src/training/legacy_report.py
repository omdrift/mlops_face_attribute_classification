"""Publish archived counts, explicitly distinguishing reused validation from a held-out test."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.splits import ATTRIBUTES


def publish_legacy_report(source="metrics/legacy_validation_metrics.json", output_dir="reports/legacy_validation"):
    original = json.loads(Path(source).read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Archived validation results", "",
             "These counts were already committed in the repository. They have not been rerun or independently verified.",
             "The old evaluator used the same `train_test_split(test_size=0.2, random_state=42)` as training and Hyperopt.",
             "Its so-called test partition was therefore the validation partition used for model selection.",
             "**These are historical validation results, not independent test performance.**", "",
             "Accuracy and macro F1 below are recalculated from the archived confusion matrices; no predictions are invented.", "",
             "| Attribute | Accuracy | Macro F1 | Support | Errors in archived matrix |",
             "|---|---:|---:|---:|---:|"]
    derived = {}
    for attr, (_, n) in ATTRIBUTES.items():
        cm = np.asarray(original[attr]["confusion_matrix"], dtype=np.int64)
        if cm.shape != (n, n) or int(cm.sum()) != original[attr]["support"]:
            raise ValueError(f"Inconsistent archived matrix: {attr}")
        tp = cm.diagonal()
        denom = cm.sum(0) + cm.sum(1)
        f1 = np.divide(2 * tp, denom, out=np.zeros(n, dtype=float), where=denom != 0)
        accuracy = float(tp.sum() / cm.sum())
        if not np.isclose(accuracy, original[attr]["accuracy"]):
            raise ValueError(f"Archived accuracy disagrees with matrix: {attr}")
        derived[attr] = {"accuracy": accuracy, "macro_f1": float(f1.mean()), "support": int(cm.sum()),
                         "errors": int(cm.sum() - tp.sum())}
        values = derived[attr]
        lines.append(f"| {attr} | {accuracy:.4f} | {f1.mean():.4f} | {values['support']} | {values['errors']} |")
        pd.DataFrame(cm, index=range(n), columns=range(n)).to_csv(output_dir / f"{attr}_confusion.csv")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set(xlabel="Predicted class index", ylabel="True class index", title=f"{attr}: archived validation",
               xticks=range(n), yticks=range(n))
        for (i, j), count in np.ndenumerate(cm):
            ax.text(j, i, str(count), ha="center", va="center", color="white" if count > cm.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(output_dir / f"{attr}_confusion.png")
        plt.close(fig)
    lines += ["", "## Missing evidence", "",
              "The original images and trained checkpoint are unavailable in this checkout. No identity annotations "
              "are present. The absence of duplicate filenames does not establish the absence of duplicate images.", "",
              "The archive has no aligned per-image true/predicted labels, so real error examples cannot be recovered "
              "from these matrices. None are fabricated.", "",
              "The committed annotations have 21,003 rows (s1: 15,890; s5: 2,613; s6: 2,500), "
              "consistent with the historical evaluation size of 4,201, but this does not prove which images were decoded.", "",
              "The former preprocessing silently skipped missing images while keeping all annotation filenames. "
              "The former training summary also saved last-epoch accuracies under the name `best_accuracies`. "
              "The archived training JSON is preserved as evidence, not presented as verified best-checkpoint metrics.", "",
              "## Confusion matrices", "", "Rows are true labels; columns are predictions. Indices follow the original CSV.", ""]
    for attr in ATTRIBUTES:
        lines += [f"### {attr}", "", f"![{attr} archived confusion matrix]({attr}_confusion.png)", ""]
    (output_dir / "README.md").write_text("\n".join(lines))
    (output_dir / "derived_metrics.json").write_text(json.dumps({"status": "archived_validation_only",
        "source": source, "independent_test": False, "attributes": derived}, indent=2) + "\n")
    return derived


if __name__ == "__main__":
    publish_legacy_report()
