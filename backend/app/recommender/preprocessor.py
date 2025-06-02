import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.recommender.data_loader import df

# prepare combined feature column
df['combined'] = df['title'] + ' ' + df['author'] + ' ' + df['genres'] + ' ' + df['description']

# tf-idf vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])

# cos sim matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# title/index helpers
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
id_to_title = dict(zip(df['bookId'], df['title']))