"""Parse influencer/post metadata and build modeling features."""

from __future__ import annotations

from pathlib import Path
import gc
import json
import re
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


TIME_BUCKETS = ("morning", "afternoon", "evening", "night", "unknown_time")
CAPTION_BUCKETS = ("short_caption", "medium_caption", "long_caption", "unknown_caption")
HASHTAG_BUCKETS = ("no_hashtags", "few_hashtags", "many_hashtags", "unknown_hashtags")
AD_BUCKETS = ("ad", "not_ad")
MEDIA_BUCKETS = ("image", "video")


def load_influencers_df(path: str | Path) -> pd.DataFrame:
    """Load and standardize influencers.txt into a dataframe."""
    path = Path(path)
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(
        columns={
            "Username": "influencer_name",
            "Category": "category",
            "#Followers": "followers",
            "#Followees": "followees",
            "#Posts": "total_posts",
        }
    )
    if "influencer_name" in df.columns:
        df = df[~df["influencer_name"].astype(str).str.startswith("=", na=False)]
    return df


def safe_get_count(data: dict[str, Any], key: str) -> int:
    value = data.get(key, {})
    if isinstance(value, dict):
        return int(value.get("count", 0) or 0)
    return 0


def extract_caption(data: dict[str, Any]) -> str:
    caption_obj = data.get("edge_media_to_caption", {})
    edges = caption_obj.get("edges", [])
    if isinstance(edges, list) and edges:
        node = edges[0].get("node", {})
        if isinstance(node, dict):
            return str(node.get("text", "") or "")
    return ""


def extract_location(data: dict[str, Any]) -> str | None:
    location = data.get("location")
    if isinstance(location, dict):
        return location.get("name")
    return None


def extract_sponsor_count(data: dict[str, Any]) -> int:
    sponsor_obj = data.get("edge_media_to_sponsor_user", {})
    if isinstance(sponsor_obj, dict):
        edges = sponsor_obj.get("edges", [])
        if isinstance(edges, list):
            return len(edges)
    return 0


def extract_hashtags(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"#(\w+)", text.lower())


def parse_influencer_from_filename(filename: str) -> str | None:
    if "-" in filename:
        return filename.rsplit("-", 1)[0]
    return None


def parse_json_name_from_filename(filename: str) -> str:
    if "-" in filename:
        return filename.rsplit("-", 1)[1]
    return filename


def parse_metadata_file_light(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
    except Exception:
        return None

    filename = path.name
    caption = extract_caption(data)
    hashtags = extract_hashtags(caption)

    owner_username = None
    if isinstance(data.get("owner"), dict):
        owner_username = data.get("owner", {}).get("username")

    influencer_from_file = parse_influencer_from_filename(filename)
    influencer_name = owner_username if owner_username else influencer_from_file

    return {
        "influencer_name": influencer_name,
        "json_postmetadata_file_name": parse_json_name_from_filename(filename),
        "extracted_file_name": filename,
        "post_id": data.get("id"),
        "shortcode": data.get("shortcode"),
        "likes": safe_get_count(data, "edge_media_preview_like"),
        "comments": safe_get_count(data, "edge_media_to_parent_comment"),
        "preview_comments": safe_get_count(data, "edge_media_preview_comment"),
        "caption": caption,
        "caption_length": len(caption) if isinstance(caption, str) else 0,
        "hashtags": hashtags,
        "hashtags_text": " ".join(hashtags),
        "num_hashtags": len(hashtags),
        "timestamp": data.get("taken_at_timestamp"),
        "is_ad": bool(data.get("is_ad", False)),
        "sponsor_count": extract_sponsor_count(data),
        "is_video": bool(data.get("is_video", False)),
        "post_type": data.get("__typename"),
        "location_name": extract_location(data),
    }


def list_metadata_files(metadata_dir: str | Path) -> list[Path]:
    metadata_dir = Path(metadata_dir)
    return sorted(
        path
        for path in metadata_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".info", ".json"}
    )


def parse_metadata_directory(
    metadata_dir: str | Path,
    batch_size: int = 1000,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Parse all metadata files in a directory, optionally writing batch parquets."""
    metadata_files = list_metadata_files(metadata_dir)
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found under {metadata_dir}")

    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for old_file in output_dir.glob("posts_part_*.parquet"):
            old_file.unlink()

    part_paths: list[Path] = []
    for start in range(0, len(metadata_files), batch_size):
        end = min(start + batch_size, len(metadata_files))
        batch_files = metadata_files[start:end]
        rows: list[dict[str, Any]] = []

        for path in tqdm(batch_files, desc=f"Parsing metadata {start}-{end}"):
            parsed = parse_metadata_file_light(path)
            if parsed is not None:
                rows.append(parsed)

        batch_df = pd.DataFrame(rows)
        if output_dir:
            part_path = output_dir / f"posts_part_{start // batch_size}.parquet"
            batch_df.to_parquet(part_path, index=False)
            part_paths.append(part_path)

        del rows, batch_df
        gc.collect()

    if output_dir and part_paths:
        return pd.concat([pd.read_parquet(path) for path in part_paths], ignore_index=True)

    rows = []
    for path in tqdm(metadata_files, desc="Parsing metadata"):
        parsed = parse_metadata_file_light(path)
        if parsed is not None:
            rows.append(parsed)
    return pd.DataFrame(rows)


def hour_bucket(hour: float) -> str:
    if pd.isna(hour):
        return "unknown_time"
    hour = int(hour)
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def caption_bucket(length: float) -> str:
    if pd.isna(length):
        return "unknown_caption"
    if length < 80:
        return "short_caption"
    if length < 200:
        return "medium_caption"
    return "long_caption"


def hashtag_bucket(count: float) -> str:
    if pd.isna(count):
        return "unknown_hashtags"
    if count == 0:
        return "no_hashtags"
    if count <= 3:
        return "few_hashtags"
    return "many_hashtags"


def build_strategy_label(row: pd.Series) -> str:
    return (
        f"{row['time_bucket']} + {row['caption_bucket']} + {row['hashtag_bucket']} + "
        f"{row['ad_bucket']} + {row['media_bucket']}"
    )


def add_strategy_features(posts_df: pd.DataFrame) -> pd.DataFrame:
    """Add time/context buckets and the combined strategy label."""
    df = posts_df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["month"] = df["datetime"].dt.month

    df["time_bucket"] = df["hour"].apply(hour_bucket)
    df["caption_bucket"] = df["caption_length"].apply(caption_bucket)
    df["hashtag_bucket"] = df["num_hashtags"].apply(hashtag_bucket)
    df["ad_bucket"] = df["is_ad"].apply(lambda value: "ad" if bool(value) else "not_ad")
    df["media_bucket"] = df["is_video"].apply(lambda value: "video" if bool(value) else "image")
    df["strategy"] = df.apply(build_strategy_label, axis=1)
    return df


def add_engagement_features(posts_df: pd.DataFrame, influencers_df: pd.DataFrame) -> pd.DataFrame:
    """Merge influencer profiles and compute engagement scores."""
    df = posts_df.merge(influencers_df, on="influencer_name", how="left")

    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
    df["comments"] = pd.to_numeric(df["comments"], errors="coerce").fillna(0)
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce")

    df = df[df["followers"].notna() & (df["followers"] > 0)].copy()

    df["raw_engagement"] = df["likes"] + 2 * df["comments"]
    df["engagement_rate"] = df["raw_engagement"] / df["followers"]
    df["log_engagement_score"] = np.log1p(df["raw_engagement"]) / np.log1p(df["followers"])
    return df


def build_posts_base(
    posts_df: pd.DataFrame,
    influencers_df: pd.DataFrame,
) -> pd.DataFrame:
    """Full preprocessing chain from parsed posts to modeling-ready table."""
    featured = add_strategy_features(posts_df)
    return add_engagement_features(featured, influencers_df)


def assign_pseudo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Assign per-influencer quintile pseudo-ratings (1-5) from engagement_rate."""
    rated = df.copy()
    rated["pseudo_rating"] = np.nan

    for _, group in rated.groupby("influencer_name"):
        if len(group) < 2:
            continue
        ranks = group["engagement_rate"].rank(method="first", pct=True)
        ratings = np.ceil(ranks * 5).clip(1, 5).astype(int)
        rated.loc[group.index, "pseudo_rating"] = ratings

    return rated
