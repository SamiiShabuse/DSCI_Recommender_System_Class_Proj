from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_import import (
    DatasetPaths,
    extract_sample_images,
    find_metadata_dir,
    list_metadata_files,
    load_influencers,
    load_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview local influencer dataset files.")
    parser.add_argument(
        "--extract-sample-images",
        action="store_true",
        help="Extract sample_images.zip into data/sample_images before previewing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = DatasetPaths(PROJECT_ROOT / "data")

    if args.extract_sample_images and dataset.sample_images_zip.exists():
        extracted = extract_sample_images(dataset.sample_images_zip, dataset.sample_images_dir)
        print(f"Extracted sample images: {len(extracted)}")

    influencers = load_influencers(dataset.influencers)
    print(f"Influencers rows: {len(influencers)}")
    print("First influencer row:")
    print(influencers[0])

    mapping = load_mapping(dataset.mapping, max_rows=5)
    print(f"Mapping preview rows: {len(mapping)}")
    print("First mapping row:")
    print(mapping[0])

    metadata_dir = find_metadata_dir(dataset)
    if metadata_dir is None:
        print("Metadata directory not found yet (expected something like data/Post_metadata).")
    else:
        metadata_preview = list_metadata_files(metadata_dir, max_files=5)
        print(f"Metadata directory found: {metadata_dir}")
        print(f"Metadata preview files: {len(metadata_preview)}")
        if metadata_preview:
            print("First metadata file:")
            print(metadata_preview[0])


if __name__ == "__main__":
    main()
