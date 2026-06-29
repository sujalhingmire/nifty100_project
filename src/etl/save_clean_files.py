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

    output_name = file.replace(".xlsx", "_clean.csv")

    df.to_csv(
        f"data/processed/{output_name}",
        index=False
    )

    print(f"Saved {output_name}")