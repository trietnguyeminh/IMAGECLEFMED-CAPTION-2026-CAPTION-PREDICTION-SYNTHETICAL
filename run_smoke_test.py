#!/usr/bin/env python3
"""Lightweight dataset-readiness smoke test for ImageCLEFmedical Caption 2026.

This script does not train or run inference. It checks whether the expected
metadata files and image folders/zip are present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def check_file(path: Path, label: str) -> bool:
    if path.exists():
        print(f"{label}: OK -> {path}")
        return True
    print(f"{label}: MISSING -> {path}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Dataset/project root")
    parser.add_argument("--captions", default="captions.csv")
    parser.add_argument("--cui-vocab", default="cui_to_name_synth_2026.csv")
    parser.add_argument("--train-concepts", default="manuel_concepts/train_concepts_manual.csv")
    parser.add_argument("--valid-concepts", default="manuel_concepts/valid_concepts_manual.csv")
    parser.add_argument("--train-images", default="train_images")
    parser.add_argument("--valid-images", default="valid_images")
    parser.add_argument("--images-zip", default="images.zip")
    args = parser.parse_args()

    root = Path(args.root)
    print("ImageCLEFmedical Caption 2026 smoke test")
    print(f"Root: {root.resolve()}")

    ok = True
    ok &= check_file(root / args.captions, "captions.csv")
    ok &= check_file(root / args.cui_vocab, "CUI vocabulary")
    ok &= check_file(root / args.train_concepts, "train concepts")
    ok &= check_file(root / args.valid_concepts, "valid concepts")

    train_dir = root / args.train_images
    valid_dir = root / args.valid_images
    images_zip = root / args.images_zip

    if train_dir.exists() and valid_dir.exists():
        print(f"image folders: OK -> {train_dir}, {valid_dir}")
    elif images_zip.exists():
        print(f"images.zip: OK -> {images_zip}")
    else:
        print("images: MISSING -> expected train/valid image folders or images.zip")
        ok = False

    captions_path = root / args.captions
    if captions_path.exists():
        df = pd.read_csv(captions_path)
        print(f"captions.csv: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    if not ok:
        print("\nSmoke test failed: dataset files are incomplete.")
        return 1

    print("\nSmoke test passed: metadata and image assets look ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
