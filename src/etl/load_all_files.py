from src.etl.loader import load_excel

files = [
    "companies.xlsx",
    "profitandloss.xlsx",
    "balancesheet.xlsx",
    "cashflow.xlsx",
    "analysis.xlsx",
    "documents.xlsx",
    "prosandcons.xlsx"
]

for file in files:
    df = load_excel(f"data/raw/{file}")
    print(file, len(df))