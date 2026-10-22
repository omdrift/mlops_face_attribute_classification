# MLOps Face Attribute Classification

Multi-head residual CNN for beard, mustache, glasses, hair color and hair length,
with a reproducible data protocol, PyTorch training, optional Hyperopt/MLflow,
DVC, a FastAPI application and Docker deployment.

[Demo](https://github.com/user-attachments/assets/b356ea14-4005-4326-9c59-d039c01841a4)

## Current evidence and limitations

**The original images are no longer available. No independent test result is claimed.**

The repository retains 21,003 annotations (15,890 from s1, 2,613 from s5 and 2,500
from s6). There are no repeated filenames, missing labels or identity columns in
this CSV. Image-content duplication and identity overlap cannot be checked from
these annotations alone.

The previous evaluator reused the validation partition used for model selection.
Its approximately 99.8% average accuracy is **historical validation performance**,
not an independent test score. The original JSON files are preserved under
`metrics/legacy_*`. [Archived per-attribute results and confusion matrices](reports/legacy_validation/README.md)
are published with these limitations. They were not rerun. No real per-image
errors can be reconstructed from aggregate counts.

The committed `data/splits/{train,validation,test}.csv` files are **provisional**:
they persist the annotation allocation, but training/evaluation refuse to use them
until the images have been audited. See [data/splits/metadata.json](data/splits/metadata.json).
[Current evaluation status](reports/evaluation_status.json) records what remains unavailable.

## Install and test without the original dataset

Python 3.10–3.12 is tested by CI. Use a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU version for tests; install a suitable CUDA build for GPU training instead.
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[test]'
python -m pip check
python -m pytest tests -q
python -m flake8 src tests --select=E9,F63,F7,F82 --show-source
```

Tests generate their own tiny images. They verify software behaviour and do not
estimate model performance. CI builds a wheel and runs the suite **outside the
checkout** against the installed package. It requires no dataset, checkpoint,
DVC credentials, Airflow installation or MLflow server.

Optional tools:

```bash
python -m pip install '.[train]'     # Hyperopt and MLflow
python -m pip install '.[pipeline]'  # DVC
```

`requirements.txt` installs both optional groups. Deployment dependencies remain
in `deployment/requirements-api.txt`; Airflow runs in its own environment.

## Freeze partitions before any new experiment

Annotations require:

```text
filename,beard,mustache,glasses_binary,hair_color_label,hair_length
```

Binary attributes use 0/1, hair color uses indices 0–4, and hair length 0–2.
Keep the original class-index mapping with the dataset; do not infer class names
from model output. Optional `identity_id`, `person_id`, `subject_id` or `identity`
columns group all images of the same person. Alternatively provide a separate
`filename,identity_id` CSV with `--identities path/to/identities.csv`. Partial
identity mappings are rejected; identities are never guessed from filenames.

If the original images are recovered, place them below `data/raw/`, then replace
the provisional manifests **before starting training**:

```bash
python -m src.data.splits --raw-dir data/raw --overwrite
# With identity metadata:
# python -m src.data.splits --raw-dir data/raw --identities identities.csv --overwrite
python -m src.data.make_dataset
```

The audit:

- Rejects missing/ambiguous/unreadable images and invalid/conflicting annotations.
- Checks SHA-256 of files, decoded pixels and the actual preprocessed model input.
- Groups near duplicates using 64-bit difference hashes (Hamming distance ≤4 by
  default) and connected components. This is a heuristic: inspect group sizes and
  known matches; it can group different images or miss transformations.
- Combines duplicate groups with supplied identity groups, so their transitive
  connections cannot cross partitions. Without identity labels, identity-disjoint
  evaluation cannot be claimed.
- Saves train/validation/test CSVs, SHA-256 fingerprints, seed, class counts,
  audit settings and a `split_id`. Defaults allocate 70/15/15 **of groups**,
  so image proportions can differ. There is no multilabel stratification; inspect
  the recorded class counts before freezing. Rare classes may be absent.

All partition files must be frozen and versioned before tuning. Do not change
seed, thresholds, labels or group assignments in response to performance.
Preparation preserves filename/label alignment and fails on changed images.
It writes separate `train.pt`, `validation.pt` and `test.pt` tensors with provenance.

To reproduce the **provisional annotation-only** allocation without images:

```bash
python -m src.data.splits --annotations-only --output-dir /tmp/provisional-splits
```

This command cannot audit image duplicates and its output cannot train a model.

## Train and select only on validation

```bash
python -m src.training.train --epochs 10 --batch-size 32
# Optional, before final training:
python -m src.training.hyperopt_search --max-evals 20 --epochs 5
python -m src.training.train --hyperopt-params src/training/hyperopt_params.json --mlflow
```

Training and Hyperopt load only the frozen train and validation tensors. The
selection criterion is the minimum validation loss: the sum of three BCE losses
and two cross-entropies. Early stopping and the scheduler consult validation.
There is no test-driven threshold tuning: binary predictions use sigmoid >0.5.
A checkpoint records its split, selection criterion, epoch, parameters and
train/validation tensor hashes. Reported best accuracies come from that saved
epoch. Historical checkpoints without this provenance are rejected for final testing.

Training writes `models/best_model.pth`, `metrics/train_metrics.json` and
`metrics/training_curves.png`. `--mlflow` enables optional local experiment tracking.
For reproducibility use the same software/hardware; a seed alone does not guarantee
identical CUDA results across platforms.

## Evaluate the reserved test once

After model/configuration selection is complete, explicitly run:

```bash
python -m src.training.evaluate
```

This creates `reports/test/` containing:

- Accuracy, macro/weighted F1, support and per-class precision/recall/F1 for each attribute.
- Confusion matrices as CSV and PNG, with fixed class indices even if a class is absent.
- Every test prediction, plus the most confident real mistakes per attribute in
  `errors.csv` and image grids (when errors exist).
- A Markdown report with the checkpoint hash, split ID and evaluation provenance.

Macro F1 includes all defined classes; an absent class contributes zero.
An exclusive access record in `data/test_access/<split_id>.json` prevents a second
run on the same split even with another checkpoint or report directory. A failure
after test access leaves a consumed record. Archive the ledger with the results;
do not delete it to compare more candidates. This is a local process safeguard,
not access control against manually bypassing the code.

The test stage is excluded from automatic DVC/Airflow retraining. New iterations
after inspecting test errors require a new, genuinely untouched holdout. A new
random split cannot undo historical exposure of these old images: the strongest
future evaluation uses a new annotated lot that was never used during development.
With the current missing images and weights, final evaluation remains **not run**.

## DVC and Airflow

```bash
dvc init                 # once; no data remote is shipped in the repository
# Audit/freeze the manifests first, then:
dvc repro train          # preparation → Hyperopt → training; no test access
dvc repro inference_batches
```

`params.yaml` controls the DVC training and Hyperopt commands. The obsolete
`dvc.lock` was removed because it described the invalid protocol; generate a new
lock only after a real run with the recovered/versioned data. DVC's existing
`data/raw.dvc` pointer alone cannot recover images without the matching cache or
remote configuration.

The notebooks refer to `/content/drive/MyDrive/S1.zip`, `S4.zip` and `S6.zip`.
These are possible recovery leads in the original Google Drive account, not
public download links. No complete image archive or checkpoint is tracked here.

Airflow deployment is in `airflow/`. Automated training and retraining DAGs do not
run the reserved test. Batch inference is separate from evaluation and has no
labelled performance claim.

## Deployment

See [deployment/Readme.md](deployment/Readme.md) for FastAPI/Docker usage.
The demo and stored prediction cache illustrate the earlier application. They do
not supply the missing labelled test images or reproduce the archived model scores.

## Repository layout

- `src/data/`: audit, persisted splits, preprocessing, tensor preparation.
- `src/training/`: training, Hyperopt, reserved-test reporting, historical reporting.
- `data/splits/`: versioned manifests and audit metadata; currently provisional.
- `metrics/legacy_*`: original metrics preserved as historical evidence.
- `reports/legacy_validation/`: published archived counts and confusion matrices.
- `tests/`: model/dataset tests and generated-fixture protocol regression tests.
- `.github/workflows/`: clean installation, tests, DVC graph checks, Docker release.
