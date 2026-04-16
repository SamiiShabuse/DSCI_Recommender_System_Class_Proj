from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_import import DatasetPaths, find_metadata_dir, list_metadata_files, load_influencers


def _top_categories(influencers: list[dict[str, object]], n: int = 10) -> list[tuple[str, int]]:
    counts = Counter(str(row.get("Category", "unknown")) for row in influencers)
    return counts.most_common(n)


def main() -> None:
    dataset = DatasetPaths(PROJECT_ROOT / "data")

    influencers = load_influencers(dataset.influencers)
    follower_values = [int(row["#Followers"]) for row in influencers]
    posts_values = [int(row["#Posts"]) for row in influencers]

    print("=== Dataset Summary ===")
    print(f"Influencer rows: {len(influencers)}")
    print(f"Unique categories: {len(set(row['Category'] for row in influencers))}")
    print(f"Followers min/avg/max: {min(follower_values)} / {sum(follower_values) // len(follower_values)} / {max(follower_values)}")
    print(f"Posts min/avg/max: {min(posts_values)} / {sum(posts_values) // len(posts_values)} / {max(posts_values)}")

    print("Top categories by influencer count:")
    for category, count in _top_categories(influencers):
        print(f"- {category}: {count}")

    if dataset.mapping.exists():
        with dataset.mapping.open("r", encoding="utf-8", errors="replace") as handle:
            mapping_rows = sum(1 for line in handle if line.strip() and not line.startswith("influencer_name") and not line.startswith("="))
        print(f"Mapping rows (approx): {mapping_rows}")

    metadata_dir = find_metadata_dir(dataset)
    if metadata_dir is None:
        print("Metadata files: not found locally yet")
    else:
        metadata_files = list_metadata_files(metadata_dir)
        print(f"Metadata directory: {metadata_dir}")
        print(f"Metadata file count: {len(metadata_files)}")
        if metadata_files:
            print(f"First metadata file: {metadata_files[0]}")

    sample_images_present = dataset.sample_images_dir.exists()
    sample_images_count = 0
    if sample_images_present:
        sample_images_count = len([p for p in dataset.sample_images_dir.rglob("*") if p.is_file()])
    print(f"Sample images extracted: {sample_images_present} ({sample_images_count} files)")


if __name__ == "__main__":
    main()
