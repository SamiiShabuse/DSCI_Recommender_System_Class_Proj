"""Weighted hybrid of collaborative filtering and content-based scores.

Model: Hybrid
What it does: Combines the scores from the collaborative filtering and content-based models.
"""

from __future__ import annotations

import pandas as pd

from src.models.content_based import ContentBasedRecommender


def normalize_scores(scores: pd.Series) -> pd.Series:
    """
    Normalizes the scores to be between 0 and 1.
    Args:
        scores: A series of scores.
    Returns:
        A series of normalized scores.
    """
    if scores.empty:
        return scores
    min_score = scores.min()
    max_score = scores.max()
    if max_score == min_score:
        return pd.Series(1.0, index=scores.index)
    return (scores - min_score) / (max_score - min_score)


def build_cf_score_series(
    influencer_name: str,
    interaction_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    n_neighbors: int = 10,
) -> pd.Series:
    """
    Builds a series of scores for each strategy by cosine similarity to the influencer's neighbors.
    Args:
        influencer_name: The name of the influencer to build scores for.
        interaction_matrix: A dataframe of interaction matrix.
        user_similarity_df: A dataframe of user similarity.
        n_neighbors: The number of neighbors to consider.
    Returns:
        A series of scores for each strategy.
    """
    if influencer_name not in interaction_matrix.index:
        return pd.Series(dtype=float)

    similarities = user_similarity_df[influencer_name].drop(index=influencer_name, errors="ignore")
    top_neighbors = similarities.sort_values(ascending=False).head(n_neighbors)
    top_neighbors = top_neighbors[top_neighbors > 0]
    if top_neighbors.empty:
        return pd.Series(dtype=float)

    neighbor_scores = interaction_matrix.loc[top_neighbors.index]
    weighted_scores = neighbor_scores.T.dot(top_neighbors)
    predicted_scores = weighted_scores / top_neighbors.sum()

    already_used = interaction_matrix.loc[influencer_name].dropna().index
    return predicted_scores.drop(index=already_used, errors="ignore")


def recommend_hybrid(
    influencer_name: str,
    interaction_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    content_recommender: ContentBasedRecommender,
    k: int = 5,
    alpha: float = 0.5,
    n_neighbors: int = 10,
) -> list[str]:
    """
    Recommends strategies by combining the scores from the collaborative filtering and content-based models.
    Args:
        influencer_name: The name of the influencer to recommend strategies for.
        interaction_matrix: A dataframe of interaction matrix.
        user_similarity_df: A dataframe of user similarity.
        content_recommender: A fitted content-based recommender.
        k: The number of strategies to recommend.
        alpha: The weight of the collaborative filtering scores.
        n_neighbors: The number of neighbors to consider.
    Returns:
        A list of recommended strategies.
    """
    cf_scores = build_cf_score_series(
        influencer_name,
        interaction_matrix,
        user_similarity_df,
        n_neighbors=n_neighbors,
    )
    content_scores = content_recommender.score_strategies(influencer_name)

    all_strategies = cf_scores.index.union(content_scores.index)
    cf_norm = normalize_scores(cf_scores.reindex(all_strategies).fillna(0))
    content_norm = normalize_scores(content_scores.reindex(all_strategies).fillna(0))

    combined = alpha * cf_norm + (1 - alpha) * content_norm
    return combined.sort_values(ascending=False).head(k).index.tolist()


def tune_hybrid_alpha(
    influencers: list[str],
    interaction_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    content_recommender: ContentBasedRecommender,
    relevance_map: dict[str, set[str]],
    k: int = 5,
    candidates: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> float:
    """
    Picks the alpha that maximizes the average hit-rate@K on a validation subset.
    Args:
        influencers: A list of influencers to tune the alpha for.
        interaction_matrix: A dataframe of interaction matrix.
        user_similarity_df: A dataframe of user similarity.
        content_recommender: A fitted content-based recommender.
        relevance_map: A dictionary of relevant strategies for each influencer.
        k: The number of strategies to recommend.
        candidates: A tuple of alpha values to try.
    Returns:
        The alpha that maximizes the average hit-rate@K.
    """
    best_alpha = 0.5
    best_score = -1.0

    for alpha in candidates:
        hits = 0
        total = 0
        for influencer in influencers:
            relevant = relevance_map.get(influencer, set())
            if not relevant:
                continue
            recs = recommend_hybrid(
                influencer,
                interaction_matrix,
                user_similarity_df,
                content_recommender,
                k=k,
                alpha=alpha,
            )
            total += 1
            if set(recs) & relevant:
                hits += 1
        if total and hits / total > best_score:
            best_score = hits / total
            best_alpha = alpha

    return best_alpha
