import pandas as pd

from src.etl.normaliser import normalize_ticker
from src.etl.normaliser import normalize_year


def load_excel(path):

    df = pd.read_excel(path, header=1)

    if "company_id" in df.columns:
        df["company_id"] = df["company_id"].apply(
            normalize_ticker
        )

    if "year" in df.columns:
        df["year"] = df["year"].apply(
            normalize_year
        )

    return df