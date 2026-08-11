"""Audit dataset metadata or build the exact-duplicate removal manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from .config import load_config
from .data import load_raw_dataset
from .utils import write_json


def viewer_json(endpoint: str, dataset_id: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"dataset": dataset_id})
    with urllib.request.urlopen(  # noqa: S310 - fixed trusted HF endpoint
        f"https://datasets-server.huggingface.co/{endpoint}?{query}", timeout=30
    ) as response:
        return json.load(response)


def image_digest(image: Image.Image) -> str:
    normalized = image.convert("RGB")
    hasher = hashlib.sha256()
    hasher.update(f"{normalized.width}x{normalized.height}:RGB".encode())
    hasher.update(normalized.tobytes())
    return hasher.hexdigest()


def difference_hash(image: Image.Image) -> str:
    pixels = list(image.convert("L").resize((9, 8)).tobytes())
    bits = []
    for row in range(8):
        start = row * 9
        bits.extend(pixels[start + col] > pixels[start + col + 1] for col in range(8))
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def metadata_audit(dataset_id: str) -> dict[str, Any]:
    return {
        "validity": viewer_json("is-valid", dataset_id),
        "splits": viewer_json("splits", dataset_id),
        "size": viewer_json("size", dataset_id),
    }


def full_audit(config_path: str, output_dir: str) -> None:
    config = load_config(config_path)
    raw = load_raw_dataset(config.data)
    output = Path(output_dir)
    exact_by_split: dict[str, dict[str, list[int]]] = {}
    perceptual_by_split: dict[str, dict[str, list[int]]] = {}
    corrupt: dict[str, list[dict[str, object]]] = {}
    dimensions: dict[str, Counter[str]] = {}
    class_counts: dict[str, Counter[int]] = {}

    for split_name in ("train", "test"):
        exact: dict[str, list[int]] = defaultdict(list)
        perceptual: dict[str, list[int]] = defaultdict(list)
        split_corrupt: list[dict[str, object]] = []
        dimension_counts: Counter[str] = Counter()
        labels: Counter[int] = Counter()
        for index, row in enumerate(tqdm(raw[split_name], desc=f"audit {split_name}")):
            try:
                image = row["image"].convert("RGB")
                exact[image_digest(image)].append(index)
                perceptual[difference_hash(image)].append(index)
                dimension_counts[f"{image.width}x{image.height}"] += 1
                labels[int(row["label"])] += 1
            except Exception as exc:  # noqa: BLE001 - audit must record any decoder failure
                split_corrupt.append({"index": index, "error": repr(exc)})
        exact_by_split[split_name] = exact
        perceptual_by_split[split_name] = perceptual
        corrupt[split_name] = split_corrupt
        dimensions[split_name] = dimension_counts
        class_counts[split_name] = labels

    train_drop: set[int] = set()
    within_train = {key: value for key, value in exact_by_split["train"].items() if len(value) > 1}
    for indices in within_train.values():
        train_drop.update(indices[1:])
    cross_split = sorted(set(exact_by_split["train"]) & set(exact_by_split["test"]))
    for digest in cross_split:
        train_drop.update(exact_by_split["train"][digest])

    manifest = {
        "dataset_id": config.data.dataset_id,
        "dataset_revision": config.data.revision,
        "policy": (
            "Keep official test copies; drop cross-split train copies "
            "and later within-train copies."
        ),
        "train_drop_indices": sorted(train_drop),
        "within_train_exact_duplicate_groups": len(within_train),
        "cross_split_exact_duplicate_groups": len(cross_split),
    }
    write_json(config.data.dedup_manifest, manifest)
    report = {
        "metadata": metadata_audit(config.data.dataset_id),
        "dataset_revision": config.data.revision,
        "dimensions": {key: dict(value) for key, value in dimensions.items()},
        "class_counts": {key: dict(value) for key, value in class_counts.items()},
        "corrupt": corrupt,
        "within_train_perceptual_hash_collision_groups": sum(
            len(indices) > 1 for indices in perceptual_by_split["train"].values()
        ),
        "within_test_perceptual_hash_collision_groups": sum(
            len(indices) > 1 for indices in perceptual_by_split["test"].values()
        ),
        "manifest": manifest,
        "note": "Perceptual hash collisions are suspects only and are not automatically removed.",
    }
    write_json(output / "dataset-audit.json", report)
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--output-dir", default="artifacts/audit")
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.full:
        full_audit(args.config, args.output_dir)
    else:
        report = metadata_audit(config.data.dataset_id)
        write_json(Path(args.output_dir) / "dataset-metadata.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
