import numpy as np
import pandas as pd

url = "https://zenodo.org/records/4265096/files/books_1.Best_Books_Ever.csv"
df = pd.read_csv(url)

# fill missing fields
df['description'] = df['description'].fillna('')
df['genres'] = df['genres'].fillna('')
df['author'] = df['author'].fillna('')
df['title'] = df['title'].fillna('')

# print(df.info())
# print(df.head())

df['popularity'] = df['rating'] * np.log1p(df['numRatings'])