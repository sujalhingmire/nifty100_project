import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.etl.loader import load_excel

df = load_excel("data/raw/profitandloss.xlsx")

print(df.head())
print(df["company_id"].head())
print(df["year"].head())