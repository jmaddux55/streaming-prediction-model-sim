from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def content_based_recommendations(x_test, x_train, y_train):
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(x_train, y_train)

    y_prob_log = log_model.predict_proba(x_test)[:, 1]

    return y_prob_log


@dataclass
class CollaborativeFiltering:
    """
    Trains a collaborative filtering model using k-NN and computes a high-resolution ROC curve.
    For each user, predicts a score for all the unrated items to improve granularity of ROC evaluation.
    i.e. treat all items not recommended as predicted low-score items (zero) and keep prediction scores unfiltered
    """

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')

    @classmethod
    def get_knn_recommendations(cls, knn_model: NearestNeighbors, user_id, user_item_matrix: DataFrame,
                                k=5, n_recommendations=100):
        if user_id not in user_item_matrix.index:
            return pd.Series(dtype='float64')

        user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
        distances, indices = knn_model.kneighbors(user_vector, n_neighbors=k+1)

        # exclude the user themselves
        neighbors = user_item_matrix.iloc[indices[0][1:]]  # skip self
        mean_ratings = neighbors.mean(axis=0)

        # filter out movies already rated by user
        rated_movies = user_item_matrix.loc[user_id]
        already_rated = rated_movies[rated_movies > 0].index
        # recommendations = mean_ratings.drop(labels=already_rated, errors='ignore')  # dropped to improve granularity

        # return recommendations.sort_values(ascending=False) #.head(n_recommendations)  # treat all items
        return mean_ratings

    @classmethod
    def evaluate_knn_accuracy(cls, user_item_matrix: DataFrame, test_df: DataFrame, k=5):
        cls.knn_model.fit(user_item_matrix)
        y_true = []
        y_score = []

        for user_id, user_group in test_df.groupby('userId'):
            if user_id not in user_item_matrix.index:
                continue

            recs = cls.get_knn_recommendations(knn_model=cls.knn_model, user_id=user_id,
                                               user_item_matrix=user_item_matrix,
                                               k=k, n_recommendations=100)

            for _, row in user_group.iterrows():
                movie_id = row['movieId']
                rating = row['rating']

                y_true.append(1 if rating >= 4 else 0)
                y_score.append(recs.get(movie_id, 0))

        if len(set(y_true)) < 2:
            return None  # cannot compute ROC AUC with only one class
        # return roc_auc_score(y_true, y_score)
        return y_true, y_score
