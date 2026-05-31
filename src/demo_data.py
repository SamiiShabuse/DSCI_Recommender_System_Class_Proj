"""Synthetic post data for local pipeline runs when metadata is unavailable."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd

from src.preprocess import (
    CAPTION_BUCKETS,
    HASHTAG_BUCKETS,
    MEDIA_BUCKETS,
    TIME_BUCKETS,
    add_engagement_features,
    add_strategy_features,
    load_influencers_df,
)


CAPTION_TEMPLATES = [
    "Loving this look today",
    "Weekend vibes and good energy",
    "New post about travel and food",
    "Training session complete",
    "Family time is the best time",
    "Product review and honest thoughts",
    "Throwback to an amazing trip",
    "Recipe idea you should try",
    "Outfit details in caption",
    "Morning coffee and planning the day",
]


def _pick_bucket(options: tuple[str, ...], rng: random.Random) -> str:
    usable = [value for value in options if not value.startswith("unknown")]
    return rng.choice(usable)


def generate_synthetic_posts(
    influencers_df: pd.DataFrame,
    num_influencers: int = 120,
    min_posts: int = 8,
    max_posts: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a realistic posts table from influencer profiles for local testing."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    eligible = influencers_df[influencers_df["followers"] >= 1000].copy()
    if len(eligible) > num_influencers:
        eligible = eligible.sample(n=num_influencers, random_state=seed)

    base_timestamp = 1_700_000_000
    rows: list[dict] = []

    for influencer_idx, influencer in eligible.iterrows():
        influencer_name = influencer["influencer_name"]
        n_posts = rng.randint(min_posts, max_posts)
        preferred_time = _pick_bucket(TIME_BUCKETS, rng)
        preferred_media = _pick_bucket(MEDIA_BUCKETS, rng)

        for post_idx in range(n_posts):
            time_bucket = preferred_time if rng.random() < 0.55 else _pick_bucket(TIME_BUCKETS, rng)
            caption_bucket = _pick_bucket(CAPTION_BUCKETS, rng)
            hashtag_bucket = _pick_bucket(HASHTAG_BUCKETS, rng)
            media_bucket = preferred_media if rng.random() < 0.6 else _pick_bucket(MEDIA_BUCKETS, rng)
            ad_bucket = "ad" if rng.random() < 0.12 else "not_ad"

            strategy = (
                f"{time_bucket} + {caption_bucket} + {hashtag_bucket} + "
                f"{ad_bucket} + {media_bucket}"
            )

            caption = f"{rng.choice(CAPTION_TEMPLATES)} #{hashtag_bucket.replace('_', '')}"
            if caption_bucket == "long_caption":
                caption += " " + " ".join(rng.choice(CAPTION_TEMPLATES) for _ in range(3))
            elif caption_bucket == "medium_caption":
                caption += " " + rng.choice(CAPTION_TEMPLATES)

            num_hashtags = {"no_hashtags": 0, "few_hashtags": rng.randint(1, 3), "many_hashtags": rng.randint(4, 8)}[
                hashtag_bucket
            ]

            followers = float(influencer["followers"])
            quality = 0.8 if ad_bucket == "not_ad" else 0.55
            if media_bucket == "video":
                quality += 0.05
            noise = float(np_rng.uniform(0.6, 1.4))

            raw_engagement = max(1.0, followers * 0.0004 * quality * noise)
            likes = int(raw_engagement * rng.uniform(0.65, 0.9))
            comments = max(0, int(raw_engagement * rng.uniform(0.05, 0.2)))

            rows.append(
                {
                    "influencer_name": influencer_name,
                    "json_postmetadata_file_name": f"synthetic_{influencer_idx}_{post_idx}.info",
                    "extracted_file_name": f"{influencer_name}-synthetic_{post_idx}.info",
                    "post_id": f"synthetic_{influencer_idx}_{post_idx}",
                    "shortcode": f"syn{influencer_idx}{post_idx}",
                    "likes": likes,
                    "comments": comments,
                    "preview_comments": comments,
                    "caption": caption,
                    "caption_length": len(caption),
                    "hashtags": [f"tag{i}" for i in range(num_hashtags)],
                    "hashtags_text": " ".join(f"tag{i}" for i in range(num_hashtags)),
                    "num_hashtags": num_hashtags,
                    "timestamp": base_timestamp + influencer_idx * 10_000 + post_idx * 86_400,
                    "is_ad": ad_bucket == "ad",
                    "sponsor_count": 1 if ad_bucket == "ad" else 0,
                    "is_video": media_bucket == "video",
                    "post_type": "GraphVideo" if media_bucket == "video" else "GraphImage",
                    "location_name": None,
                    "strategy": strategy,
                    "time_bucket": time_bucket,
                    "caption_bucket": caption_bucket,
                    "hashtag_bucket": hashtag_bucket,
                    "ad_bucket": ad_bucket,
                    "media_bucket": media_bucket,
                }
            )

    posts_df = pd.DataFrame(rows)
    posts_df["datetime"] = pd.to_datetime(posts_df["timestamp"], unit="s", errors="coerce")
    posts_df["hour"] = posts_df["datetime"].dt.hour
    posts_df["day_of_week"] = posts_df["datetime"].dt.day_name()
    posts_df["month"] = posts_df["datetime"].dt.month
    return posts_df


def build_synthetic_posts_base(
    influencers_path: str | Path,
    num_influencers: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    influencers_df = load_influencers_df(influencers_path)
    posts_df = generate_synthetic_posts(
        influencers_df,
        num_influencers=num_influencers,
        seed=seed,
    )
    return add_engagement_features(posts_df, influencers_df)


def build_posts_base_from_parquet_or_synthetic(
    influencers_path: str | Path,
    posts_parquet: str | Path | None = None,
    synthetic: bool = False,
    num_influencers: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    influencers_df = load_influencers_df(influencers_path)

    if posts_parquet:
        posts_df = pd.read_parquet(posts_parquet)
        if "strategy" not in posts_df.columns:
            posts_df = add_strategy_features(posts_df)
        return add_engagement_features(posts_df, influencers_df)

    if synthetic:
        return build_synthetic_posts_base(influencers_path, num_influencers=num_influencers, seed=seed)

    raise ValueError("Provide --posts-parquet, --extracted-metadata-dir, or --synthetic.")
