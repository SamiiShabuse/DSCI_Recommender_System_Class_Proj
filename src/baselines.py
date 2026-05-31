"""Popularity and category baseline recommenders.

Model: Global baseline
What it does: Ranks strategies by average engagement across all influencers.

Model: Category baseline
What it does: Ranks strategies by average engagement within each influencer category.
"""

from __future__ import annotations

import pandas as pd

def build_strategy_scores(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a dataframe of strategy scores by averaging the log_engagement_score for each strategy.
    Args:
        train_df: A dataframe of training data.
    Returns:
        A dataframe of strategy scores.
    """
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
    """
    Builds a dataframe of strategy scores by averaging the log_engagement_score for each strategy within each category.
    Args:
        train_df: A dataframe of training data.
    Returns:
        A dataframe of strategy scores.
    """
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


def recommend_global_strategies(strategy_scores: pd.DataFrame, k: int = 5, min_posts: int = 10) -> list[str]:
    """
    Recommends strategies by average engagement across all influencers.
    Args:
        strategy_scores: A dataframe of strategy scores.
        k: The number of strategies to recommend.
        min_posts: The minimum number of posts for a strategy to be recommended.
    Returns:
        A list of recommended strategies.
    """
    filtered = strategy_scores[strategy_scores["post_count"] >= min_posts]
    ranked = filtered.sort_values("avg_engagement", ascending=False)
    return ranked["strategy"].head(k).tolist()


def recommend_by_category(category: str, category_strategy_scores: pd.DataFrame, k: int = 5, min_posts: int = 3) -> list[str]:
    """
    Recommends strategies by average engagement within each influencer category.
    Args:
        category: The category to recommend strategies for.
        category_strategy_scores: A dataframe of strategy scores by category.
        k: The number of strategies to recommend.
        min_posts: The minimum number of posts for a strategy to be recommended.
    Returns:
        A list of recommended strategies.
    """
    filtered = category_strategy_scores[(category_strategy_scores["category"] == category) & (category_strategy_scores["post_count"] >= min_posts)]
    ranked = filtered.sort_values("avg_engagement", ascending=False)
    return ranked["strategy"].head(k).tolist()

def recommend_global_for_influencer(influencer_name: str, train_df: pd.DataFrame, strategy_scores: pd.DataFrame, k: int = 5, min_posts: int = 10) -> list[str]:
    """
    Recommends strategies by average engagement across all influencers.
    Args:
        influencer_name: The name of the influencer to recommend strategies for.
        train_df: A dataframe of training data.
        strategy_scores: A dataframe of strategy scores.
        k: The number of strategies to recommend.
        min_posts: The minimum number of posts for a strategy to be recommended.
    Returns:
        A list of recommended strategies.
    """
    del influencer_name, train_df
    return recommend_global_strategies(strategy_scores, k=k, min_posts=min_posts)


def recommend_category_for_influencer(
    influencer_name: str,
    train_df: pd.DataFrame,
    category_strategy_scores: pd.DataFrame,
    k: int = 5,
    min_posts: int = 3,
) -> list[str]:
    """
    Recommends strategies by average engagement within each influencer category.
    Args:
        influencer_name: The name of the influencer to recommend strategies for.
        train_df: Training posts used to look up the influencer's category.
        category_strategy_scores: A dataframe of strategy scores by category.
        k: The number of strategies to recommend.
        min_posts: The minimum number of posts for a strategy to be recommended.
    Returns:
        A list of recommended strategies.
    """
    user_rows = train_df[train_df["influencer_name"] == influencer_name]
    if user_rows.empty:
        return []
    category = user_rows["category"].iloc[0]
    if pd.isna(category):
        return []
    return recommend_by_category(str(category), category_strategy_scores, k=k, min_posts=min_posts)
