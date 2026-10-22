"""Create persistent, auditable partitions without consulting model performance."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessing import PREPROCESS_VERSION, array_hash, preprocess_image

ATTRIBUTES = {
    "beard": ("beard", 2),
    "mustache": ("mustache", 2),
    "glasses": ("glasses_binary", 2),
    "hair_color": ("hair_color_label", 5),
    "hair_length": ("hair_length", 3),
}
LABEL_COLUMNS = [column for column, _ in ATTRIBUTES.values()]
PARTITIONS = ("train", "validation", "test")
HASH_COLUMNS = ("sha256", "pixel_sha256", "input_sha256")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_annotations(df):
    required = ["filename", *LABEL_COLUMNS]
    if any(column not in df for column in required):
        raise ValueError(f"Annotations must contain {required}")
    if df.empty or df[required].isna().any().any():
        raise ValueError("Annotations must be nonempty, with no missing filenames or labels")
    for name in df.filename:
        path = Path(str(name))
        if not str(name).strip() or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe or empty filename: {name}")
    for column, classes in ATTRIBUTES.values():
        if not df[column].isin(range(classes)).all():
            raise ValueError(f"Invalid labels in {column}; expected 0..{classes - 1}")


def image_index(raw_dir):
    index = {}
    for path in sorted(Path(raw_dir).rglob("*")):
        if path.is_file():
            index.setdefault(path.name, []).append(path)
    return index


def resolve_image(filename, raw_dir, index):
    direct = Path(raw_dir) / filename
    if direct.is_file():
        return direct
    candidates = index.get(Path(filename).name, [])
    if len(candidates) != 1:
        raise ValueError(f"Missing or ambiguous image: {filename} ({len(candidates)} matches)")
    return candidates[0]


def difference_hash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")


class HashTree:
    """BK tree for Hamming-distance lookup, avoiding an all-pairs image scan."""
    def __init__(self):
        self.root = None

    def find_and_add(self, value, row, radius):
        if self.root is None:
            self.root = (value, row, {})
            return []
        matches, pending = [], [self.root]
        while pending:
            other, index, children = pending.pop()
            distance = (value ^ other).bit_count()
            if distance <= radius:
                matches.append(index)
            pending.extend(node for edge, node in children.items() if abs(edge - distance) <= radius)
        node = self.root
        while True:
            other, _, children = node
            distance = (value ^ other).bit_count()
            if distance == 0:
                break
            if distance not in children:
                children[distance] = (value, row, {})
                break
            node = children[distance]
        return matches


def create_splits(annotations, raw_dir="data/raw", output_dir="data/splits", *,
                  seed=42, validation_size=0.15, test_size=0.15, identity_column=None,
                  identities=None, annotations_only=False, near_threshold=4, overwrite=False):
    if not (0 < validation_size < 1 and 0 < test_size < 1 and validation_size + test_size < 1):
        raise ValueError("Train, validation and test fractions must all be positive")
    if not 0 <= near_threshold <= 8:
        raise ValueError("near_threshold must be between 0 and 8")
    output_dir = Path(output_dir)
    if any((output_dir / name).exists() for name in ["metadata.json", *[f"{p}.csv" for p in PARTITIONS]]):
        if not overwrite:
            raise FileExistsError("Partitions already exist; freeze them or explicitly use --overwrite before training")
    identity_names = ["identity_id", "person_id", "subject_id", "identity"]
    if identity_column:
        identity_names.append(identity_column)
    df = pd.read_csv(annotations, dtype={c: str for c in ["filename", *identity_names]})
    validate_annotations(df)
    if identities:
        mapping = pd.read_csv(identities, dtype=str)
        if not {"filename", "identity_id"}.issubset(mapping.columns):
            raise ValueError("Identity CSV must contain filename,identity_id")
        if "identity_id" in df:
            raise ValueError("Supply identity_id in annotations or a mapping, not both")
        df = df.merge(mapping[["filename", "identity_id"]], on="filename", how="left", validate="many_to_one")
        identity_column = "identity_id"
    if identity_column is None:
        identity_column = next((c for c in ("identity_id", "person_id", "subject_id", "identity") if c in df), None)
    if identity_column:
        if identity_column not in df or df[identity_column].isna().any():
            raise ValueError("Identity information must cover every annotated image")
        df["identity_id"] = df[identity_column].astype(str).str.strip()
        if df.identity_id.eq("").any():
            raise ValueError("Identity information must not be empty")
    check_columns = LABEL_COLUMNS + (["identity_id"] if identity_column else [])
    if (df.groupby("filename")[check_columns].nunique() > 1).any().any():
        raise ValueError("Conflicting annotations for the same filename")
    duplicate_rows = int(df.duplicated("filename").sum())
    df = df.drop_duplicates("filename").sort_values("filename").reset_index(drop=True)
    df[LABEL_COLUMNS] = df[LABEL_COLUMNS].astype(int)
    parent = list(range(len(df)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)

    audit = {"images_checked": not annotations_only, "duplicate_annotation_rows_removed": duplicate_rows,
             "identity_separated": bool(identity_column), "identity_column": identity_column,
             "near_duplicate_hamming_threshold": near_threshold if not annotations_only else None}
    if not annotations_only:
        index, tree = image_index(raw_dir), HashTree()
        records = []
        for i, row in df.iterrows():
            path = resolve_image(row.filename, raw_dir, index)
            img = cv2.imread(str(path))
            processed = preprocess_image(img)
            perceptual = difference_hash(img)
            records.append([file_hash(path), array_hash(img), array_hash(processed), f"{perceptual:016x}"])
            for j in tree.find_and_add(perceptual, i, near_threshold):
                union(i, j)
        df[[*HASH_COLUMNS, "dhash"]] = records
        for column in HASH_COLUMNS:
            if (df.groupby(column)[LABEL_COLUMNS].nunique() > 1).any().any():
                raise ValueError(f"Identical image content has conflicting labels ({column})")
            audit[f"{column}_duplicate_groups"] = int((df.groupby(column).size() > 1).sum())
    for column in (["identity_id"] if identity_column else []) + ([] if annotations_only else list(HASH_COLUMNS)):
        for indices in df.groupby(column).indices.values():
            for i in indices[1:]:
                union(int(indices[0]), int(i))
    df["group_id"] = [df.iloc[root(i)].filename for i in range(len(df))]
    groups = sorted(df.group_id.unique())
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required for train/validation/test")
    train_val, test = train_test_split(groups, test_size=test_size, random_state=seed)
    train, validation = train_test_split(train_val, test_size=validation_size / (1 - test_size), random_state=seed)
    partitions = {name: df[df.group_id.isin(group)].copy() for name, group in
                  zip(PARTITIONS, (train, validation, test))}
    metadata = {"protocol_version": 1, "status": "provisional" if annotations_only else "audited",
                "annotations_sha256": file_hash(annotations), "seed": seed,
                "requested_fractions": {"train": 1 - validation_size - test_size,
                                        "validation": validation_size, "test": test_size},
                "allocation": "random groups; fractions apply to groups, no label stratification",
                "preprocess_version": PREPROCESS_VERSION, "audit": audit,
                "groups": len(groups), "samples": len(df), "partitions": {}}
    payloads = {name: part.to_csv(index=False, lineterminator="\n").encode() for name, part in partitions.items()}
    for name, part in partitions.items():
        metadata["partitions"][name] = {
            "size": len(part), "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            "class_counts": {attr: {str(k): int((part[col] == k).sum()) for k in range(n)}
                             for attr, (col, n) in ATTRIBUTES.items()},
        }
    metadata["split_id"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (output_dir / f"{name}.csv").write_bytes(payload)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_splits(directory="data/splits", *, require_audit=True):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    unsigned = {k: v for k, v in metadata.items() if k != "split_id"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != metadata["split_id"]:
        raise ValueError("Split metadata fingerprint mismatch")
    if require_audit and (metadata["status"] != "audited" or not metadata["audit"]["images_checked"]):
        raise ValueError("Provisional partitions: original images are required to audit duplicates before training")
    partitions = {}
    seen = {c: set() for c in ["filename", "group_id", "identity_id", *HASH_COLUMNS]}
    for name in PARTITIONS:
        path = directory / f"{name}.csv"
        if file_hash(path) != metadata["partitions"][name]["sha256"]:
            raise ValueError(f"Modified partition: {name}")
        df = pd.read_csv(path, dtype={"filename": str, "identity_id": str})
        validate_annotations(df)
        if len(df) != metadata["partitions"][name]["size"] or df.filename.duplicated().any():
            raise ValueError(f"Invalid partition size or duplicate filename: {name}")
        for col in seen:
            if col in df:
                values = set(df[col])
                if values & seen[col]:
                    raise ValueError(f"Leakage across partitions: {col}")
                seen[col].update(values)
        partitions[name] = df
    return partitions, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/annotations/mapped_train.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--identity-column")
    parser.add_argument("--identities", help="Optional filename,identity_id CSV")
    parser.add_argument("--annotations-only", action="store_true", help="Provisional manifests; training will refuse them")
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    metadata = create_splits(**vars(parser.parse_args()))
    print(json.dumps({"status": metadata["status"], "split_id": metadata["split_id"],
                      "sizes": {k: v["size"] for k, v in metadata["partitions"].items()}}, indent=2))


if __name__ == "__main__":
    main()
