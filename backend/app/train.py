import os
import joblib
from surprise import SVD, Dataset, Reader
from services.db_service import get_all_interactions


def train_and_save():
    interactions = get_all_interactions()
    if interactions.empty:
        return

    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(interactions[['user_id', 'book_id', 'liked']], reader)
    trainset = data.build_full_trainset()

    model = SVD()
    model.fit(trainset)

    joblib.dump(model, "backend/app/models/svd_model.pkl")

if __name__ == "__main__":
    train_and_save()