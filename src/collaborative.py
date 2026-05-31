"""User-based collaborative filtering recommender.

Model: User-based collaborative filtering
What it does: Ranks strategies by cosine similarity between influencers on strategy engagement vectors.

"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_interaction_matrix(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a dataframe of interaction matrix by averaging the log_engagement_score for each strategy.
    Args:
        train_df: A dataframe of training data.
    Returns:
        A dataframe of interaction matrix.
    """
    return train_df.pivot_table(
        index="influencer_name",
        columns="strategy",
        values="log_engagement_score",
        aggfunc="mean",
    )


def build_user_similarity(interaction_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a dataframe of user similarity by cosine similarity between influencers on 
    strategy engagement vectors.
    Args:
        interaction_matrix: A dataframe of interaction matrix.
    Returns:
        A dataframe of user similarity.
    """
    filled = interaction_matrix.fillna(0)
    similarity = cosine_similarity(filled)
    return pd.DataFrame(similarity, index=filled.index, columns=filled.index)


def recommend_user_based_cf(
    influencer_name: str,
    interaction_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    k: int = 5,
    n_neighbors: int = 10,
) -> list[str]:
    """
    Recommends strategies by cosine similarity between influencers on strategy engagement vectors.
    Args:
        influencer_name: The name of the influencer to recommend strategies for.
        interaction_matrix: A dataframe of interaction matrix.
        user_similarity_df: A dataframe of user similarity.
        k: The number of strategies to recommend.
        n_neighbors: The number of neighbors to consider.
    Returns:
        A list of recommended strategies.
    """
    if influencer_name not in interaction_matrix.index:
        return []

    similarities = user_similarity_df[influencer_name].drop(index=influencer_name, errors="ignore")
    top_neighbors = similarities.sort_values(ascending=False).head(n_neighbors)
    top_neighbors = top_neighbors[top_neighbors > 0]
    if top_neighbors.empty:
        return []

    neighbor_scores = interaction_matrix.loc[top_neighbors.index]
    weighted_scores = neighbor_scores.T.dot(top_neighbors)
    predicted_scores = weighted_scores / top_neighbors.sum()

    already_used = interaction_matrix.loc[influencer_name].dropna().index
    predicted_scores = predicted_scores.drop(index=already_used, errors="ignore")

    return predicted_scores.sort_values(ascending=False).head(k).index.tolist()


def show_influencer_history(influencer_name: str, posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shows the history of an influencer's posts.
    Args:
        influencer_name: The name of the influencer to show the history for.
        posts_df: A dataframe of posts.
    Returns:
        A dataframe of the influencer's history.
    """
    return (
        posts_df[posts_df["influencer_name"] == influencer_name][
            [
                "influencer_name",
                "category",
                "likes",
                "comments",
                "log_engagement_score",
                "strategy",
            ]
        ]
        .sort_values("log_engagement_score", ascending=False)
    )
