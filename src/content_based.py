"""Caption TF-IDF content-based strategy recommender.

Model: Content-based
What it does: TF-IDF on captions to recommend strategies similar to what worked for that influencer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ContentBasedRecommender:
    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            min_df=2,
            ngram_range=(1, 2),
        )
    )
    strategy_vectors_: pd.DataFrame | None = None
    train_df_: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> ContentBasedRecommender:
        self.train_df_ = train_df.copy()
        captions = train_df["caption"].fillna("").astype(str)
        self.vectorizer.fit(captions)

        strategy_vectors: dict[str, np.ndarray] = {}
        for strategy, group in train_df.groupby("strategy"):
            indices = group.index
            matrix = self.vectorizer.transform(group["caption"].fillna("").astype(str))
            weights = group["log_engagement_score"].to_numpy()
            if weights.sum() <= 0:
                weights = np.ones(len(weights))
            weighted = matrix.multiply(weights[:, None]).mean(axis=0)
            strategy_vectors[strategy] = np.asarray(weighted).ravel()

        self.strategy_vectors_ = pd.DataFrame(strategy_vectors).T
        self.strategy_vectors_.columns = [f"v{i}" for i in range(self.strategy_vectors_.shape[1])]
        return self

    def _user_profile(self, influencer_name: str) -> np.ndarray | None:
        assert self.train_df_ is not None
        user_posts = self.train_df_[self.train_df_["influencer_name"] == influencer_name]
        if user_posts.empty:
            return None

        matrix = self.vectorizer.transform(user_posts["caption"].fillna("").astype(str))
        weights = user_posts["log_engagement_score"].to_numpy()
        if weights.sum() <= 0:
            weights = np.ones(len(weights))
        profile = matrix.multiply(weights[:, None]).mean(axis=0)
        return np.asarray(profile).ravel()

    def score_strategies(self, influencer_name: str) -> pd.Series:
        if self.strategy_vectors_ is None or self.train_df_ is None:
            raise RuntimeError("ContentBasedRecommender must be fit before scoring.")

        profile = self._user_profile(influencer_name)
        if profile is None:
            return pd.Series(dtype=float)

        similarities = cosine_similarity(
            profile.reshape(1, -1),
            self.strategy_vectors_.to_numpy(),
        ).ravel()

        scores = pd.Series(similarities, index=self.strategy_vectors_.index)
        used = set(
            self.train_df_.loc[
                self.train_df_["influencer_name"] == influencer_name,
                "strategy",
            ]
        )
        return scores.drop(index=list(used), errors="ignore").sort_values(ascending=False)

    def recommend(self, influencer_name: str, k: int = 5) -> list[str]:
        scores = self.score_strategies(influencer_name)
        return scores.head(k).index.tolist()


def recommend_content_based(
    influencer_name: str,
    recommender: ContentBasedRecommender,
    k: int = 5,
) -> list[str]:
    return recommender.recommend(influencer_name, k=k)
