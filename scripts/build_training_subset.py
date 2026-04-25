from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_import import DatasetPaths, find_metadata_dir, load_influencers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a manageable influencer/post subset for model development in resource-limited "
            "environments like Google Colab."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory that contains influencers.txt, JSON-Image_files_mapping.txt, and metadata folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "processed",
        help="Directory where subset files will be written.",
    )
    parser.add_argument(
        "--num-influencers",
        type=int,
        default=1500,
        help="How many influencers to sample for the training subset.",
    )
    parser.add_argument(
        "--min-followers",
        type=int,
        default=5000,
        help="Only sample influencers with at least this many followers.",
    )
    parser.add_argument(
        "--max-posts-per-influencer",
        type=int,
        default=150,
        help="Upper bound on mapping rows kept per influencer.",
    )
    parser.add_argument(
        "--max-metadata-files",
        type=int,
        default=None,
        help="Optional hard cap on metadata files loaded after mapping selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def choose_influencers(
    influencers: list[dict[str, Any]],
    num_influencers: int,
    min_followers: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [row for row in influencers if int(row["#Followers"]) >= min_followers]
    if not eligible:
        raise ValueError(
            f"No influencers match min_followers={min_followers}. Lower the threshold and retry."
        )

    random_generator = random.Random(seed)
    if len(eligible) <= num_influencers:
        selected = eligible
    else:
        selected = random_generator.sample(eligible, k=num_influencers)

    selected.sort(key=lambda row: int(row["#Followers"]), reverse=True)
    return selected


def parse_mapping_line(line: str, line_number: int) -> tuple[str, str, list[str]]:
    parts = line.split(maxsplit=2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected mapping format on line {line_number}: {line[:120]}")

    influencer_name, metadata_file, image_list_text = parts
    image_files = ast.literal_eval(image_list_text)
    if not isinstance(image_files, list):
        raise ValueError(f"Expected list of image files on line {line_number}")

    return influencer_name, metadata_file, image_files


def parse_metadata_payload(file_path: Path) -> Any:
    raw_text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw_text:
        return {}

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(raw_text)
    except (ValueError, SyntaxError):
        return {"raw_text": raw_text}


def build_selected_mapping(
    mapping_path: Path,
    output_csv_path: Path,
    selected_usernames: set[str],
    max_posts_per_influencer: int,
) -> tuple[set[str], int, int, int]:
    metadata_files_needed: set[str] = set()
    kept_rows = 0
    skipped_bad_rows = 0
    read_rows = 0
    post_counts: dict[str, int] = {}

    with mapping_path.open("r", encoding="utf-8", errors="replace") as source, output_csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as sink:
        writer = csv.DictWriter(
            sink,
            fieldnames=[
                "influencer_name",
                "json_postmetadata_file_name",
                "image_count",
            ],
        )
        writer.writeheader()

        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line or line.startswith("influencer_name") or line.startswith("="):
                continue

            read_rows += 1
            try:
                influencer_name, metadata_file, image_files = parse_mapping_line(line, line_number)
            except ValueError:
                skipped_bad_rows += 1
                continue

            if influencer_name not in selected_usernames:
                continue

            current_count = post_counts.get(influencer_name, 0)
            if current_count >= max_posts_per_influencer:
                continue

            post_counts[influencer_name] = current_count + 1
            metadata_files_needed.add(metadata_file)
            kept_rows += 1

            writer.writerow(
                {
                    "influencer_name": influencer_name,
                    "json_postmetadata_file_name": metadata_file,
                    "image_count": len(image_files),
                }
            )

    return metadata_files_needed, read_rows, kept_rows, skipped_bad_rows


def build_selected_metadata_jsonl(
    metadata_dir: Path,
    metadata_files_needed: set[str],
    output_jsonl_path: Path,
) -> tuple[int, int]:
    matched_files = 0
    missing_files = 0
    remaining = set(metadata_files_needed)

    with output_jsonl_path.open("w", encoding="utf-8") as sink:
        for file_path in metadata_dir.rglob("*"):
            if not remaining:
                break
            if not file_path.is_file() or file_path.suffix.lower() not in {".info", ".json"}:
                continue

            file_name = file_path.name
            if file_name not in remaining:
                continue

            payload = parse_metadata_payload(file_path)
            sink.write(
                json.dumps(
                    {
                        "metadata_file_name": file_name,
                        "metadata_path": str(file_path),
                        "payload": payload,
                    }
                )
                + "\n"
            )
            remaining.remove(file_name)
            matched_files += 1

    missing_files = len(remaining)
    return matched_files, missing_files


def main() -> None:
    args = parse_args()

    dataset = DatasetPaths(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading influencer table...")
    influencers = load_influencers(dataset.influencers)

    print(
        f"Selecting up to {args.num_influencers} influencers with >= {args.min_followers} followers "
        f"(seed={args.seed})..."
    )
    selected_influencers = choose_influencers(
        influencers=influencers,
        num_influencers=args.num_influencers,
        min_followers=args.min_followers,
        seed=args.seed,
    )
    selected_usernames = {str(row["Username"]) for row in selected_influencers}

    selected_influencers_path = args.output_dir / "selected_influencers.csv"
    with selected_influencers_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Username", "Category", "#Followers", "#Followees", "#Posts"],
        )
        writer.writeheader()
        writer.writerows(selected_influencers)

    print("Selecting mapping rows for sampled influencers...")
    selected_mapping_path = args.output_dir / "selected_mapping.csv"
    metadata_files_needed, read_rows, kept_rows, bad_rows = build_selected_mapping(
        mapping_path=dataset.mapping,
        output_csv_path=selected_mapping_path,
        selected_usernames=selected_usernames,
        max_posts_per_influencer=args.max_posts_per_influencer,
    )

    if args.max_metadata_files is not None and args.max_metadata_files > 0:
        metadata_files_needed = set(sorted(metadata_files_needed)[: args.max_metadata_files])

    metadata_dir = find_metadata_dir(dataset)
    selected_metadata_path = args.output_dir / "selected_metadata.jsonl"

    if metadata_dir is None:
        print("Metadata directory not found; skipped metadata extraction step.")
        matched_files = 0
        missing_files = len(metadata_files_needed)
    else:
        print("Extracting selected metadata records...")
        matched_files, missing_files = build_selected_metadata_jsonl(
            metadata_dir=metadata_dir,
            metadata_files_needed=metadata_files_needed,
            output_jsonl_path=selected_metadata_path,
        )

    print("\n=== Subset Build Complete ===")
    print(f"Selected influencers: {len(selected_influencers)}")
    print(f"Mapping rows scanned: {read_rows}")
    print(f"Mapping rows kept: {kept_rows}")
    print(f"Malformed mapping rows skipped: {bad_rows}")
    print(f"Metadata files requested: {len(metadata_files_needed)}")
    print(f"Metadata files matched: {matched_files}")
    print(f"Metadata files missing: {missing_files}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
