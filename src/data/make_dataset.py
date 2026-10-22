"""Prepare separate tensors from audited, frozen CSV manifests."""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.dataset import ImageDataset
from src.data.preprocessing import PREPROCESS_VERSION, array_hash, preprocess_image
from src.data.splits import LABEL_COLUMNS, PARTITIONS, file_hash, image_index, load_splits, resolve_image


def tensor_digest(images, labels):
    return hashlib.sha256((array_hash(images.cpu().numpy()) + array_hash(labels.cpu().numpy())).encode()).hexdigest()


def process_training_data(raw_dir="data/raw", splits_dir="data/splits", output_dir="data/processed"):
    partitions, metadata = load_splits(splits_dir)
    index = image_index(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in partitions.items():
        images = []
        for row in df.itertuples(index=False):
            path = resolve_image(row.filename, raw_dir, index)
            if file_hash(path) != row.sha256:
                raise ValueError(f"Image changed after duplicate audit: {row.filename}")
            image = preprocess_image(cv2.imread(str(path)))
            if array_hash(image) != row.input_sha256:
                raise ValueError(f"Preprocessing changed after duplicate audit: {row.filename}")
            images.append(image)
        x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
        y = torch.from_numpy(df[LABEL_COLUMNS].to_numpy(dtype=np.int64))
        bundle = {"X": x, "y": y, "filenames": df.filename.tolist(), "partition": name,
                  "split_id": metadata["split_id"], "preprocess_version": PREPROCESS_VERSION,
                  "tensors_sha256": tensor_digest(x, y)}
        temporary = output_dir / f"{name}.pt.tmp"
        torch.save(bundle, temporary)
        temporary.replace(output_dir / f"{name}.pt")
    return metadata


def load_partition(name, processed_dir="data/processed", splits_dir="data/splits"):
    if name not in PARTITIONS:
        raise ValueError(f"Unknown partition: {name}")
    partitions, metadata = load_splits(splits_dir)
    bundle = torch.load(Path(processed_dir) / f"{name}.pt", map_location="cpu", weights_only=True)
    frame = partitions[name]
    if bundle.get("split_id") != metadata["split_id"] or bundle.get("partition") != name:
        raise ValueError("Processed partition does not match the frozen split")
    if bundle.get("preprocess_version") != PREPROCESS_VERSION:
        raise ValueError("Preprocessing version mismatch")
    if bundle["filenames"] != frame.filename.tolist() or not np.array_equal(bundle["y"].numpy(), frame[LABEL_COLUMNS]):
        raise ValueError("Misaligned filenames or labels in processed data")
    if bundle.get("tensors_sha256") != tensor_digest(bundle["X"], bundle["y"]):
        raise ValueError("Processed tensors have been modified")
    if len(bundle["X"]) != len(frame) or not torch.isfinite(bundle["X"]).all():
        raise ValueError("Invalid image tensors")
    return ImageDataset(bundle["X"], bundle["y"], is_preprocessed=True), bundle, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--output-dir", default="data/processed")
    process_training_data(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
