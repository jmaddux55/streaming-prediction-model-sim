from datetime import datetime

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from pathlib import Path
from rapidfuzz import process, fuzz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from Models.systems import content_based_recommendations, CollaborativeFiltering
from profile import User, Movie, MovieRating

save_new_movie_titles = False


def recommend_movies(user_id, user_similarities, n_recommendations=5):
    # get similar scores for the target user
    # (userId zero-based, similarity score)
    sim_scores = list(enumerate(user_similarities[user_id - 1]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # find the most similar user
    most_similar_user = sim_scores[1][0] + 1  # account for 0-based index

    # movies rated by the similar user
    similar_user_movies = ratings_df[ratings_df['userId'] == most_similar_user]
    similar_user_movies = similar_user_movies.sort_values(by='rating', ascending=False)

    # filter out movies already seen by target user
    user_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()

    # recommend movies that similar user liked that target user hasn't seen
    recommendations = similar_user_movies[~similar_user_movies['movieId'].isin(user_movies)].head(n_recommendations)

    return movies_df[movies_df['movieId'].isin(recommendations['movieId'])]


def convert_date(date):
    if date is not None:
        return datetime.strptime(date, "%Y-%m-%d")
    else:
        return None


def extract_title_year(title):
    match = re.match(r"^(.*)\s+\((\d{4})\)$", title)
    if match:
        clean_title = match.group(1).strip()
        year = match.group(2)
        return clean_title, year
    else:
        return title.strip(), None


def extract_title_info(title):
    aka_year_pattern = r"^(.*?)\s*\((?:a\.k\.a\.|aka)\s*(.*?)\)\s*\((\d{4})\)$"
    match = re.match(aka_year_pattern, title, flags=re.IGNORECASE)
    if match:
        return {
            'main_title': match.group(1).strip(),
            'alt_title': match.group(2).strip(),
            'year': match.group(3)
        }

    aka_only_pattern = r"^(.*?)\s*\((?:a\.k\.a\.|aka)\s*(.?)\)$"
    match = re.match(aka_only_pattern, title, flags=re.IGNORECASE)
    if match:
        return {
            'main_title': match.group(1).strip(),
            'alt_title': match.group(2).strip(),
            'year': None
        }

    title_year_pattern = r"^(.*)\s+\((\d{4})\)$"
    match = re.match(title_year_pattern, title)
    if match:
        return {
            'main_title': match.group(1).strip(),
            'alt_title': None,
            'year': match.group(2)
        }

    return {
        'main_title': title.strip(),
        'alt_title': None,
        'year': None
    }


def match_title(title: str, match_list: list, yr_list: list):
    matches = process.extract(title, match_list, scorer=fuzz.token_sort_ratio, limit=None)

    if len(matches) == 0:
        raise AttributeError(f"Failed to find a title for '{title}'")

    elif len(matches) == 1:
        print(f"Found a match for '{title}' ~= '{matches[0][0]}' {yr_list[matches[0][2]]} (score: {matches[0][1]}")
        return matches[0][0]

    else:
        clean_title, yr_check = extract_title_year(title)
        if yr_check is not None:
            for i, (match, score, yr) in enumerate(matches):
                if yr_check in yr_list[yr]:
                    print(f"Found a match for '{title}' ~= '{match}' {yr_list[yr]} (score: {score}")
                    return match

        print(f"\nChoose a match for '{title}'")
        for i, (match, score, yr) in enumerate(matches):
            print(f"{i + 1}: {match} {yr_list[yr]} (score: {score})")

        user_input = input(f"Choose a match for '{title}' or press Enter to skip:  ").strip()
        if user_input.isdigit():
            idx = int(user_input)
            return matches[idx - 1][0]
        return None


def replace_movie_title(movie_title: str, analytics: DataFrame):
    # title is exact match
    lk_row = analytics.loc[analytics['title'] == movie_title]

    # movie_title contains the year or alt title
    if lk_row.empty:
        matched = False
        # lk_row = analytics.loc[analytics['title'].str.contains(movie_title, case=False, na=False)]
        # clean_title, yr_check = extract_title_year(movie_title)
        clean_title, alt_title, yr_check = extract_title_info(movie_title)
        lk_row = analytics.loc[analytics['title'] == clean_title]
        if not lk_row.empty:
            if yr_check in lk_row['release_date'].values[0]:
                matched = True
        else:
            lk_row = analytics.loc[analytics['title'] == alt_title]
            if not lk_row.empty:
                if yr_check in lk_row['release_date'].values[0]:
                    matched = True

        # ask user for the best match
        if not matched:
            match = match_title(movie_title, analytics['title'].tolist(), analytics['release_date'].tolist())
            if match is None:
                return movie_title
            lk_row = analytics.loc[analytics['title'] == match]

    return lk_row['title'].values[0]


def lookup_movie_analytics(movie_title: str, analytics: DataFrame):
    """return: id (int), genres (List[str]), releaseDate (datetime), popularity (float), keywords (List[str])"""

    # title is exact match
    lk_row = analytics.loc[analytics['title'] == movie_title]

    # movie_title contains the year or alt title
    if lk_row.empty:
        matched = False
        # lk_row = analytics.loc[analytics['title'].str.contains(movie_title, case=False, na=False)]
        # clean_title, yr_check = extract_title_year(movie_title)
        clean_title, alt_title, yr_check = extract_title_info(movie_title)
        lk_row = analytics.loc[analytics['title'] == clean_title]
        if not lk_row.empty:
            if yr_check in lk_row['release_date'].values[0]:
                matched = True
        else:
            lk_row = analytics.loc[analytics['title'] == alt_title]
            if not lk_row.empty:
                if yr_check in lk_row['release_date'].values[0]:
                    matched = True

        # ask user for the best match
        if not matched:
            match = match_title(movie_title, analytics['title'].tolist(), analytics['release_date'].tolist())
            if match is None:
                return None
            lk_row = analytics.loc[analytics['title'] == match]

    date = convert_date(lk_row['release_date'].values[0]) if lk_row['release_date'].values[0] != '' else ''

    # return [int(lk_row['id'].values[0]), lk_row['genres'].values[0].split('-'),
    #         date, float(lk_row['popularity'].values[0]), lk_row['keywords'].values[0].split('-') ]
    return [int(lk_row['id'].values[0]), lk_row['genres'].values[0].replace('-', '|'),
            date, float(lk_row['popularity'].values[0]), lk_row['keywords'].values[0].replace('-', '|')]


dataset = Path(__file__).parent.joinpath('ml-latest-small')
ratings_df = pd.read_csv(dataset / 'ratings.csv')
if save_new_movie_titles:
    movie_analytics_df = pd.read_csv(dataset / 'Movies Daily Update Dataset export 2025-04-28 21-26-24.csv')
    movie_analytics_df.fillna('', inplace=True)
    movies_df = pd.read_csv(dataset / 'movies.csv')
    # movies_df['matched_title'] = movies_df['title'].apply(replace_movie_title, analytics=movie_analytics_df)
    merged_df = movies_df.merge(movie_analytics_df[['title', 'popularity', 'keywords', 'release_date']],
                                left_on='matched_title', right_on='title', how='left')
    cleaned_df = merged_df.drop_duplicates(subset='movieId', keep='first').reset_index(drop=True)
    cleaned_df.to_csv(dataset / f"updated_movies.csv", index=False)
else:
    movies_df = pd.read_csv(dataset / 'updated_movies.csv')
    merged_df = ratings_df.merge(movies_df, on='movieId')
    merged_df['genres'] = merged_df['genres'].str.split('|')

    # we are only going to care about the year this time
    merged_df['release_date'] = merged_df['release_date'].apply(
        lambda x: convert_date(str(x)).year if not pd.isna(x) else x
    )
    merged_df['release_date'] = merged_df['release_date'].apply(
        lambda x: 0 if pd.isna(x) else x
    )

    # fill empty keywords with empty list and split out the keywords
    merged_df['keywords'] = merged_df['keywords'].apply(
        lambda x: [] if pd.isna(x) or x == '' else x
    )
    merged_df['keywords'] = merged_df['keywords'].apply(
        lambda x: x.split('-') if isinstance(x, str) else x
    )

filtered_df = merged_df.fillna(0)

# binarize genres and keywords for ML user
# coverts to a binary matrix for easier processing
mlb_genres = MultiLabelBinarizer()
mlb_keywords = MultiLabelBinarizer()

# expand features
# Fit the label sets binarizer and transform the given label sets
genres_encoded = mlb_genres.fit_transform(filtered_df['genres'])
keywords_encoded = mlb_keywords.fit_transform(filtered_df['keywords'])

# combine features
X_features = np.hstack([
    filtered_df[['popularity', 'release_date']].values,
    genres_encoded,
    keywords_encoded
])

# define binary target: liked (rating >= 4) vs not
y = (filtered_df['rating'] >= 4).astype(int)

# split data
# Split arrays or matrices into random train and test subsets.
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.1, random_state=42)

# --------- Logistic Regression (Content-Based) ---------
# y_prob_log = content_based_recommendations(X_test, X_train, y_train)
# np.save(dataset / 'y_prob_log.npy', y_prob_log)
y_prob_log = np.load(dataset / 'y_prob_log.npy')

# false positive (user didn't rate) & true positive (user did rate)
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
auc_log = auc(fpr_log, tpr_log)

# --------- Collaborative Filtering (k-NN on ratings) ---------
# Split arrays or matrices into random train and test subsets.
X_baseline = merged_df[['userId', 'movieId', 'popularity', 'release_date', 'genres', 'keywords']]
X_baseline = pd.get_dummies(X_baseline.astype(str))
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
# train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)
# train_df, test_df = train_test_split(X_baseline, test_size=0.2, random_state=42)

# sort the pivot table for each user-movie-rating
train_agg = train_df.groupby(['userId', 'movieId'])['rating'].mean().reset_index()
user_item_matrix_knn = train_agg.pivot(index='userId', columns='movieId', values='rating').fillna(0)

cf = CollaborativeFiltering()
y_true, y_score = cf.evaluate_knn_accuracy(user_item_matrix=user_item_matrix_knn, test_df=test_df, k=100)
y_score = np.array(y_score)
if y_score.max() > y_score.min():
    y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
if len(set(y_true)) < 2:
    raise ValueError("ROC AUC cannot be  computed - only one class present in y_true")

fpr_knn, tpr_knn, _ = roc_curve(y_true, y_score)
auc_knn = auc(fpr_knn, tpr_knn)

# --------- Plot ROC Curves ---------
plt.figure(figsize=(10, 6))
plt.plot(fpr_log, tpr_log, label=f"Content Based (LogReg) AUC = {auc_log:.2f}")
plt.plot(fpr_knn, tpr_knn, label=f'Collaborative Filtering (Simulated k-NN) AUC = {auc_knn:.2}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Recommendation Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

# abandon object based logic for the robustness of DataFrames
"""
# cosine similarities between users
# similarity score of row user and column user (zero-based userId)
user_similarities = cosine_similarity(sparse_matrix)

print(recommend_movies(user_id=1, user_similarities=user_similarities, n_recommendations=5))
print('')


movies = {}
for _, row in movies_df.iterrows():
    m_analytics = lookup_movie_analytics(row['title'], movie_analytics_df)
    if m_analytics is not None:
        movies[row['movieId']] = Movie(dbId=m_analytics[0],
                                       title=row['title'],
                                       genre=m_analytics[1],
                                       releaseDate=m_analytics[2],
                                       popularity=m_analytics[3]
                                       )

users = {}
for _, row in ratings_df.iterrows():
    if row['userId'] not in users:
        users[row['userId']] = User(userId=row['userId'],
                                    ratings={},
                                    genres={}
                                    )

    if row['title'] in movies:
        users[row['userId']].rate_movie(movie=movies[row['title']],
                                        rating=MovieRating(movieId=row['movieId'],
                                                           rating=row['rating'],
                                                           timestamp=datetime.fromtimestamp(int(row['timestamp']
                                                                                                         )
                                                                                                     )
                                                           )
                                        )

print(users[1].top_genres())
print(users[1].avg_rating())
"""


def plot_precision_recall_and_score_distribution(y_true, y_score, label='Model'):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f'{label} (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(y_score, bins=50, alpha=0.7, label='Predicted scores')
    plt.xlabel('Predicted Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=True)


plot_precision_recall_and_score_distribution(y_true, y_score, label='k-NN Collaborative')


# Baseline

X_baseline = merged_df[['userId', 'movieId']]
X_baseline = pd.get_dummies(X_baseline.astype(str))
y = (merged_df['rating'] >= 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_baseline, y, test_size=0.1, random_state=42)

y_prob_base = content_based_recommendations(X_test, X_train, y_train)
# np.save(dataset / 'y_prob_log.npy', y_prob_log)
# y_prob_log = np.load(dataset / 'y_prob_log.npy')

# false positive (user didn't rate) & true positive (user did rate)
fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
auc_base = auc(fpr_base, fpr_base)

plt.figure(figsize=(10, 6))
plt.plot(fpr_log, tpr_log, label=f"Content Based (LogReg) AUC = {auc_log:.2f}")
plt.plot(fpr_base, tpr_base, label=f'Post-Correlation Content Based (LogReg) AUC = {auc_base:.2}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Recommendation Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

plot_precision_recall_and_score_distribution(y_test, y_prob_log, label='Content Based (LogReg)')
plot_precision_recall_and_score_distribution(y_test, y_prob_base, label='Post-Correlation Content Based (LogReg)')

"""
def compare_logistic_regression_feature_sets(ratings_df, movies_df):
    ratings_df['liked'] = (ratings_df['rating'] >= 4).astype(int)

    df = ratings_df.merge(movies_df, on='movieId', how='left')

    df['genres'] = df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    df['keywords'] = df['keywords'].apply(lambda x: x.split('-') if isinstance(x, str) else [])
"""
