"""Popularity and category baseline recommenders."""

from __future__ import annotations

import pandas as pd


def build_strategy_scores(train_df: pd.DataFrame) -> pd.DataFrame:
    return (
        train_df.groupby("strategy")
        .agg(
            post_count=("strategy", "count"),
            avg_engagement=("log_engagement_score", "mean"),
            avg_likes=("likes", "mean"),
            avg_comments=("comments", "mean"),
        )
        .reset_index()
        .sort_values("avg_engagement", ascending=False)
    )


def build_category_strategy_scores(train_df: pd.DataFrame) -> pd.DataFrame:
    return (
        train_df.groupby(["category", "strategy"])
        .agg(
            post_count=("strategy", "count"),
            avg_engagement=("log_engagement_score", "mean"),
            avg_likes=("likes", "mean"),
            avg_comments=("comments", "mean"),
        )
        .reset_index()
        .sort_values(["category", "avg_engagement"], ascending=[True, False])
    )


def recommend_global_strategies(
    strategy_scores: pd.DataFrame,
    k: int = 5,
    min_posts: int = 10,
) -> list[str]:
    filtered = strategy_scores[strategy_scores["post_count"] >= min_posts]
    ranked = filtered.sort_values("avg_engagement", ascending=False)
    return ranked["strategy"].head(k).tolist()


def recommend_by_category(
    category: str,
    category_strategy_scores: pd.DataFrame,
    k: int = 5,
    min_posts: int = 3,
) -> list[str]:
    filtered = category_strategy_scores[
        (category_strategy_scores["category"] == category)
        & (category_strategy_scores["post_count"] >= min_posts)
    ]
    ranked = filtered.sort_values("avg_engagement", ascending=False)
    return ranked["strategy"].head(k).tolist()


def recommend_global_for_influencer(
    influencer_name: str,
    train_df: pd.DataFrame,
    strategy_scores: pd.DataFrame,
    k: int = 5,
    min_posts: int = 10,
) -> list[str]:
    del influencer_name, train_df
    return recommend_global_strategies(strategy_scores, k=k, min_posts=min_posts)


def recommend_category_for_influencer(
    influencer_name: str,
    train_df: pd.DataFrame,
    category_strategy_scores: pd.DataFrame,
    k: int = 5,
    min_posts: int = 3,
) -> list[str]:
    user_rows = train_df[train_df["influencer_name"] == influencer_name]
    if user_rows.empty:
        return []
    category = user_rows["category"].iloc[0]
    if pd.isna(category):
        return []
    return recommend_by_category(str(category), category_strategy_scores, k=k, min_posts=min_posts)
