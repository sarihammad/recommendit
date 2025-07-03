import json
import pandas as pd

def parse_json_list_field(field):
    """
    Parses a JSON-like string field and returns a concatenated string
    of the 'name' attributes inside the JSON objects.
    """
    try:
        sanitized = field.replace('""', '"')
        items = json.loads(sanitized)
        return " ".join(item.get("name", "") for item in items)
    except Exception:
        return ""

def load_movie_data(path="data/movies.csv"):
    """
    Loads movie data from CSV and preprocesses fields for recommendation.
    """
    df = pd.read_csv(path)

    df = df[["title", "overview", "genres", "keywords", "release_date"]].dropna()

    # parse JSON fields
    df["genres"] = df["genres"].apply(parse_json_list_field)
    df["keywords"] = df["keywords"].apply(parse_json_list_field)

    # combine the relevant text fields for recommendation
    df["combined"] = (
        df["title"] + " " +
        df["overview"] + " " +
        df["genres"] + " " +
        df["keywords"]
    ).str.lower()

    return df