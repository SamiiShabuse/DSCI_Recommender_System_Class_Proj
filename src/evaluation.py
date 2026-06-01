"""Train/test splits and ranking metrics for recommender evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from src.models.baselines import (
    build_category_strategy_scores,
    build_strategy_scores,
    recommend_category_for_influencer,
    recommend_global_for_influencer,
)
from src.models.collaborative import build_interaction_matrix, build_user_similarity, recommend_user_based_cf
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import recommend_hybrid, tune_hybrid_alpha
from src.data.preprocess import assign_pseudo_ratings


def time_based_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_train_posts: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent posts per influencer for testing."""
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    sort_col = "timestamp" if "timestamp" in df.columns else "datetime"
    for _, group in df.sort_values(sort_col).groupby("influencer_name"):
        if len(group) <= min_train_posts:
            train_parts.append(group)
            continue

        split_idx = max(1, int(len(group) * (1 - test_ratio)))
        if split_idx >= len(group):
            split_idx = len(group) - 1

        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_df, test_df


def relevant_strategies_from_test(
    test_user_df: pd.DataFrame,
    min_rating: int = 5,
) -> set[str]:
    rated = test_user_df.dropna(subset=["pseudo_rating"])
    if rated.empty:
        return set()
    return set(rated.loc[rated["pseudo_rating"] >= min_rating, "strategy"].unique())


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if not relevant:
        return 0.0
    return len(set(top_k) & relevant) / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    top_k = recommended[:k]
    if not relevant:
        return 0.0
    return len(set(top_k) & relevant) / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    top_k = recommended[:k]
    if not relevant or not top_k:
        return 0.0

    dcg = 0.0
    for index, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / np.log2(index + 2)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    top_k = recommended[:k]
    return 1.0 if set(top_k) & relevant else 0.0


def evaluate_recommender(
    recommend_fn: Callable[[str], list[str]],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 5,
    min_rating: int = 5,
) -> dict[str, float]:
    metrics = defaultdict(list)

    for influencer, test_group in test_df.groupby("influencer_name"):
        if influencer not in set(train_df["influencer_name"]):
            continue

        relevant = relevant_strategies_from_test(test_group, min_rating=min_rating)
        if not relevant:
            continue

        recommended = recommend_fn(influencer)
        if not recommended:
            continue

        metrics["precision"].append(precision_at_k(recommended, relevant, k))
        metrics["recall"].append(recall_at_k(recommended, relevant, k))
        metrics["ndcg"].append(ndcg_at_k(recommended, relevant, k))
        metrics["hit_rate"].append(hit_rate_at_k(recommended, relevant, k))

    if not metrics["precision"]:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "evaluated_users": 0,
        }

    return {
        "precision_at_k": float(np.mean(metrics["precision"])),
        "recall_at_k": float(np.mean(metrics["recall"])),
        "ndcg_at_k": float(np.mean(metrics["ndcg"])),
        "hit_rate_at_k": float(np.mean(metrics["hit_rate"])),
        "evaluated_users": len(metrics["precision"]),
    }


def build_model_artifacts(train_df: pd.DataFrame) -> dict:
    strategy_scores = build_strategy_scores(train_df)
    category_strategy_scores = build_category_strategy_scores(train_df)
    interaction_matrix = build_interaction_matrix(train_df)
    user_similarity_df = build_user_similarity(interaction_matrix)
    content_recommender = ContentBasedRecommender().fit(train_df)
    return {
        "strategy_scores": strategy_scores,
        "category_strategy_scores": category_strategy_scores,
        "interaction_matrix": interaction_matrix,
        "user_similarity_df": user_similarity_df,
        "content_recommender": content_recommender,
    }


def evaluate_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 5,
    min_rating: int = 5,
    hybrid_alpha: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Evaluate global, category, CF, content-based, and hybrid models."""
    artifacts = build_model_artifacts(train_df)

    if hybrid_alpha is None:
        relevance_map = {
            influencer: relevant_strategies_from_test(group, min_rating=min_rating)
            for influencer, group in test_df.groupby("influencer_name")
        }
        val_users = [
            user
            for user, rel in relevance_map.items()
            if rel and user in artifacts["interaction_matrix"].index
        ]
        if val_users:
            hybrid_alpha = tune_hybrid_alpha(
                val_users[: min(25, len(val_users))],
                artifacts["interaction_matrix"],
                artifacts["user_similarity_df"],
                artifacts["content_recommender"],
                relevance_map,
                k=k,
            )
        else:
            hybrid_alpha = 0.5

    model_fns = {
        "global_baseline": lambda user: recommend_global_for_influencer(
            user,
            train_df,
            artifacts["strategy_scores"],
            k=k,
        ),
        "category_baseline": lambda user: recommend_category_for_influencer(
            user,
            train_df,
            artifacts["category_strategy_scores"],
            k=k,
        ),
        "user_based_cf": lambda user: recommend_user_based_cf(
            user,
            artifacts["interaction_matrix"],
            artifacts["user_similarity_df"],
            k=k,
        ),
        "content_based": lambda user: artifacts["content_recommender"].recommend(user, k=k),
        "hybrid": lambda user: recommend_hybrid(
            user,
            artifacts["interaction_matrix"],
            artifacts["user_similarity_df"],
            artifacts["content_recommender"],
            k=k,
            alpha=hybrid_alpha,
        ),
    }

    rows = []
    for model_name, recommend_fn in model_fns.items():
        result = evaluate_recommender(recommend_fn, train_df, test_df, k=k, min_rating=min_rating)
        rows.append({"model": model_name, "k": k, "hybrid_alpha": hybrid_alpha, **result})

    return pd.DataFrame(rows), hybrid_alpha


def prepare_evaluation_frames(posts_base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rated = assign_pseudo_ratings(posts_base_df)
    return time_based_split(rated)
