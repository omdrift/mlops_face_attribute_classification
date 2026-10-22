"""Regression tests for data leakage, provenance and reserved-test access.

All images here are generated fixtures, never published as research results.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.make_dataset import load_partition, process_training_data
from src.data.splits import ATTRIBUTES, HashTree, create_splits, load_splits
from src.training.evaluate import attribute_metrics, evaluate_model
from src.training.train import run_training


@pytest.fixture
def image_corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng, rows = np.random.default_rng(9), []
    for i in range(20):
        name = f"s1_{i:05}.png"
        cv2.imwrite(str(raw / name), rng.integers(0, 240, size=(64, 64, 3), dtype=np.uint8))
        rows.append({"filename": name, "identity_id": f"subject-{i // 2}", "beard": i % 2,
                     "mustache": (i // 2) % 2, "glasses_binary": i % 2,
                     "hair_color_label": i % 5, "hair_length": i % 3})
    csv = tmp_path / "annotations.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return raw, csv, tmp_path / "splits", tmp_path / "processed"


def audited_corpus(image_corpus):
    raw, csv, splits, processed = image_corpus
    create_splits(csv, raw, splits, near_threshold=0)
    process_training_data(raw, splits, processed)
    return raw, csv, splits, processed


def test_stable_persisted_partitions_and_identity_isolation(image_corpus, tmp_path):
    raw, csv, splits, _ = image_corpus
    first = create_splits(csv, raw, splits, near_threshold=0)
    frames, _ = load_splits(splits)
    assert sum(len(df) for df in frames.values()) == 20
    seen = set()
    for df in frames.values():
        assert not seen.intersection(df.identity_id)
        seen.update(df.identity_id)
    pd.read_csv(csv).sample(frac=1, random_state=11).to_csv(csv, index=False)
    second_dir = tmp_path / "reordered"
    create_splits(csv, raw, second_dir, near_threshold=0)
    for name in frames:
        assert (splits / f"{name}.csv").read_bytes() == (second_dir / f"{name}.csv").read_bytes()
    with pytest.raises(FileExistsError):
        create_splits(csv, raw, splits)
    (splits / "test.csv").write_text((splits / "test.csv").read_text().replace("s1_", "s2_", 1))
    with pytest.raises(ValueError, match="Modified partition"):
        load_splits(splits)
    assert first["audit"]["identity_separated"]


def test_duplicate_content_bridges_identity_groups(image_corpus):
    raw, csv, splits, _ = image_corpus
    df = pd.read_csv(csv)
    shutil.copyfile(raw / df.iloc[0].filename, raw / df.iloc[2].filename)
    for col in [value[0] for value in ATTRIBUTES.values()]:
        df.loc[2, col] = df.loc[0, col]
    df.to_csv(csv, index=False)
    metadata = create_splits(csv, raw, splits, near_threshold=0)
    frames, _ = load_splits(splits)
    # Duplicate file connects two identities: all four images must stay together.
    assert any(set(df.filename[:4]).issubset(set(frame.filename)) for frame in frames.values())
    assert metadata["audit"]["sha256_duplicate_groups"] == 1


def test_conflicting_duplicate_content_is_rejected(image_corpus):
    raw, csv, splits, _ = image_corpus
    df = pd.read_csv(csv)
    shutil.copyfile(raw / df.iloc[0].filename, raw / df.iloc[1].filename)
    with pytest.raises(ValueError, match="conflicting labels"):
        create_splits(csv, raw, splits)
    assert not (splits / "metadata.json").exists()


def test_annotations_only_is_honestly_provisional(image_corpus):
    raw, csv, splits, processed = image_corpus
    df = pd.read_csv(csv).drop(columns="identity_id")
    df.to_csv(csv, index=False)
    shutil.rmtree(raw)
    meta = create_splits(csv, raw, splits, annotations_only=True)
    assert not meta["audit"]["images_checked"]
    assert not meta["audit"]["identity_separated"]
    frames, _ = load_splits(splits, require_audit=False)
    assert len(set.union(*(set(df.filename) for df in frames.values()))) == 20
    with pytest.raises(ValueError, match="Provisional partitions"):
        process_training_data(raw, splits, processed)


def test_missing_and_ambiguous_images_do_not_silently_shift_labels(image_corpus):
    raw, csv, splits, _ = image_corpus
    name = pd.read_csv(csv).iloc[0].filename
    image = cv2.imread(str(raw / name))
    (raw / name).unlink()
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        create_splits(csv, raw, splits)
    for folder in (raw / "one", raw / "two"):
        folder.mkdir()
        cv2.imwrite(str(folder / name), image)
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        create_splits(csv, raw, splits)


def test_partial_identities_and_invalid_labels_are_rejected(image_corpus):
    raw, csv, splits, _ = image_corpus
    df = pd.read_csv(csv)
    df.loc[0, "identity_id"] = None
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Identity information"):
        create_splits(csv, raw, splits)
    df.loc[0, "identity_id"] = "subject-0"
    df["beard"] = df["beard"].astype(float)
    df.loc[0, "beard"] = 0.5
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Invalid labels"):
        create_splits(csv, raw, splits)


def test_processed_filenames_labels_and_content_are_checked(image_corpus):
    _, _, splits, processed = audited_corpus(image_corpus)
    path = processed / "validation.pt"
    bundle = torch.load(path, weights_only=True)
    bundle["filenames"].reverse()
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="Misaligned"):
        load_partition("validation", processed, splits)
    bundle["filenames"].reverse()
    bundle["X"][0, 0, 0, 0] += 0.1
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="modified"):
        load_partition("validation", processed, splits)


def test_bktree_matches_brute_force():
    tree, previous = HashTree(), []
    rng = np.random.default_rng(10)
    for i, value in enumerate(rng.integers(0, 2 ** 16, size=100).tolist()):
        actual = tree.find_and_add(value, i, 4)
        # Exact duplicate keys have one representative; union transitivity handles repeats.
        expected = [j for j, old in previous if (value ^ old).bit_count() <= 4]
        assert sorted(actual) == sorted(expected)
        if all(old != value for _, old in previous):
            previous.append((i, value))


def test_absent_classes_keep_fixed_confusion_matrix_axes():
    true = {attr: [0, 0] for attr in ATTRIBUTES}
    pred = {attr: [0, 1] for attr in ATTRIBUTES}
    result = attribute_metrics(true, pred)
    assert result["hair_color"]["confusion_matrix"] == [[1, 1, 0, 0, 0], [0] * 5, [0] * 5, [0] * 5, [0] * 5]
    assert result["beard"]["accuracy"] == 0.5
    assert result["beard"]["macro_f1"] == pytest.approx(1 / 3)


def test_training_never_opens_test_and_saves_best_epoch_metrics(image_corpus, tmp_path, monkeypatch):
    _, _, splits, processed = audited_corpus(image_corpus)
    # Training must work even when the reserved test tensors are inaccessible.
    (processed / "test.pt").unlink()
    import src.training.train as training
    values = iter([(2.0, {k: 0.8 for k in ATTRIBUTES}), (3.0, {k: 0.2 for k in ATTRIBUTES})])
    monkeypatch.setattr(training, "validate", lambda *args: next(values))
    model_path = tmp_path / "model.pth"
    result = run_training(processed_dir=processed, splits_dir=splits, model_path=model_path,
                          metrics_path=tmp_path / "metrics.json", epochs=2, batch_size=4)
    assert result["best_epoch"] == 1
    assert result["best_accuracies"]["beard"] == 0.8
    checkpoint = torch.load(model_path, weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["selection_partition"] == "validation"
    assert result["history"][-1]["validation_accuracy"]["beard"] == 0.2


def test_complete_generated_fixture_pipeline_and_single_test_use(image_corpus, tmp_path):
    _, _, splits, processed = audited_corpus(image_corpus)
    model_path, output, ledger = tmp_path / "model.pth", tmp_path / "report", tmp_path / "ledger"
    run_training(processed_dir=processed, splits_dir=splits, model_path=model_path,
                 metrics_path=tmp_path / "train.json", epochs=1, batch_size=4)
    kwargs = dict(processed_dir=processed, splits_dir=splits, model_path=model_path,
                  output_dir=output, ledger_dir=ledger, batch_size=2, max_errors=2)
    result = evaluate_model(**kwargs)
    assert result["provenance"]["status"] == "completed"
    assert result["provenance"]["identity_separated"]
    frame = pd.read_csv(output / "predictions.csv")
    assert len(frame) == result["attributes"]["beard"]["support"]
    for attr in ATTRIBUTES:
        assert result["attributes"][attr]["accuracy"] == (frame[f"true_{attr}"] == frame[f"pred_{attr}"]).mean()
        assert (output / f"{attr}_confusion.png").exists()
    for row in pd.read_csv(output / "errors.csv").itertuples():
        assert row.true != row.predicted
        assert row.filename in frame.filename.values
    assert (output / "README.md").is_file()
    # A different output directory must not allow evaluating another candidate on the same test.
    kwargs["output_dir"] = tmp_path / "second-report"
    with pytest.raises(RuntimeError, match="already accessed"):
        evaluate_model(**kwargs)
    assert not kwargs["output_dir"].exists()


def test_legacy_checkpoint_cannot_be_claimed_as_new_test(image_corpus, tmp_path):
    _, _, splits, processed = audited_corpus(image_corpus)
    path = tmp_path / "legacy.pth"
    torch.save({"model_state_dict": {}}, path)
    ledger = tmp_path / "ledger"
    with pytest.raises(ValueError, match="legacy models are rejected"):
        evaluate_model(processed_dir=processed, splits_dir=splits, model_path=path,
                       output_dir=tmp_path / "report", ledger_dir=ledger)
    assert not ledger.exists()


def test_hyperopt_does_not_open_test_tensors(image_corpus, tmp_path):
    from src.training.hyperopt_search import run_hyperopt
    _, _, splits, processed = audited_corpus(image_corpus)
    (processed / "test.pt").unlink()
    result = run_hyperopt(max_evals=1, epochs=1, processed_dir=processed, splits_dir=splits,
                         output=tmp_path / "hyperopt.json")
    assert result["selection_partition"] == "validation"
    assert result["selection_metric"] == "validation_loss"
    assert result["params"]["batch_size"] in (32, 64)
    run_training(processed_dir=processed, splits_dir=splits, epochs=1,
                 hyperopt_params=tmp_path / "hyperopt.json", model_path=tmp_path / "tuned.pth",
                 metrics_path=tmp_path / "tuned-metrics.json")


def test_failure_after_test_access_does_not_allow_comparing_another_model(image_corpus, tmp_path, monkeypatch):
    import src.training.evaluate as evaluation
    _, _, splits, processed = audited_corpus(image_corpus)
    model_path, ledger = tmp_path / "model.pth", tmp_path / "ledger"
    run_training(processed_dir=processed, splits_dir=splits, model_path=model_path,
                 metrics_path=tmp_path / "train.json", epochs=1)

    def failed_report(*args, **kwargs):
        raise OSError("simulated report write failure")

    monkeypatch.setattr(evaluation, "publish_report", failed_report)
    kwargs = dict(processed_dir=processed, splits_dir=splits, model_path=model_path,
                  output_dir=tmp_path / "failed-report", ledger_dir=ledger)
    with pytest.raises(OSError, match="simulated"):
        evaluate_model(**kwargs)
    record = json.loads(next(ledger.glob("*.json")).read_text())
    assert record["status"] == "failed_after_test_access"
    kwargs["output_dir"] = tmp_path / "retry"
    with pytest.raises(RuntimeError, match="already accessed"):
        evaluate_model(**kwargs)


def test_separate_identity_mapping_preserves_leading_zeros(image_corpus, tmp_path):
    raw, csv, splits, _ = image_corpus
    frame = pd.read_csv(csv)
    mapping = tmp_path / "identities.csv"
    pd.DataFrame({"filename": frame.filename, "identity_id": [f"{i // 2:03}" for i in range(len(frame))]}).to_csv(mapping, index=False)
    frame.drop(columns="identity_id").to_csv(csv, index=False)
    create_splits(csv, raw, splits, identities=mapping, near_threshold=0)
    parts, meta = load_splits(splits)
    assert meta["audit"]["identity_separated"]
    assert set.union(*(set(p.identity_id) for p in parts.values())) == {f"{i:03}" for i in range(10)}
