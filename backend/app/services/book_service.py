import joblib
from sklearn.preprocessing import MinMaxScaler
from app.recommender.preprocessor import df, tfidf_matrix, cosine_sim, indices, id_to_title
from app.recommender.model_loader import load_model
import pandas as pd

svd_model = load_model()

def recommend_hybrid(user_id, liked_book_ids, top_n=5):
    # if new user with no feedback, return popular books
    if not liked_book_ids:
        return df.sort_values(by='popularity', ascending=False).head(top_n)[['title', 'author', 'genres', 'rating']]

    # adjust weights based on user's level of interaction. Low interaction: purely content-based filtering. High interaction: shift to collaborative filtering.
    if len(liked_book_ids) < 3:
        w_cf = 0.0
        w_cb = 1.0
    else:
        w_cf = 0.6
        w_cb = 0.4

    # get collaborative predictions
    unseen_books = df[~df['bookId'].isin(liked_book_ids)].copy()
    unseen_books['cf_score'] = unseen_books['bookId'].apply(
        lambda book_id: svd_model.predict(user_id, book_id).est if svd_model else 0
    )

    # normalize collaborative scores
    scaler_cf = MinMaxScaler()
    unseen_books['cf_score_norm'] = scaler_cf.fit_transform(unseen_books[['cf_score']])

    # content-based: generate recs and assign score using rank penalty (1 / (rank + 1))
    content_recs = recommend_from_likes(liked_book_ids, top_n=top_n * 2)
    content_scores = {
        row['title']: 1 / (i + 1)
        for i, row in content_recs.iterrows()
    }

    # apply content-based rank penalty as score (or default low value)
    unseen_books['cb_score'] = unseen_books['title'].apply(
        lambda t: content_scores.get(t, 0)
    )

    # normalize content-based scores
    scaler_cb = MinMaxScaler()
    unseen_books['cb_score_norm'] = scaler_cb.fit_transform(unseen_books[['cb_score']])

    # final hybrid score using weighted sum
    unseen_books['hybrid_score'] = (
        w_cf * unseen_books['cf_score_norm'] +
        w_cb * unseen_books['cb_score_norm']
    )

    return unseen_books.sort_values(by='hybrid_score', ascending=False).head(top_n)[['title', 'author', 'genres', 'rating']]


def top_books(n=5):
    return df.sort_values(by='numRatings', ascending=False).head(n)[['title', 'author', 'genres', 'rating']]

def recommend_from_likes(book_ids, top_n=5):
    liked_titles = [id_to_title.get(bid) for bid in book_ids if bid in id_to_title and id_to_title.get(bid)]
    idxs = [indices.get(title.lower()) for title in liked_titles if indices.get(title.lower()) is not None]

    if not idxs:
        return pd.DataFrame()

    avg_vector = tfidf_matrix[idxs].mean(axis=0)
    sim_scores = cosine_sim(avg_vector, tfidf_matrix).flatten()
    sim_scores = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)
    book_indices = [i[0] for i in sim_scores if i[0] not in idxs][:top_n]

    return df.iloc[book_indices][['title', 'author', 'genres', 'rating']]


def recommend_collaboratively(user_id, top_n=5):
    if not svd_model:
        return pd.DataFrame()

    book_ids = df['bookId'].unique()
    predictions = [
        (book_id, svd_model.predict(user_id, book_id).est)
        for book_id in book_ids
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_books_ids = [bid for bid, _ in predictions[:top_n]]

    return df[df['bookId'].isin(top_books_ids)][['title', 'author', 'genres', 'rating']]