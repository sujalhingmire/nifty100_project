import sqlite3
import pandas as pd

def check_ratios_data():
    conn = sqlite3.connect('data/nifty100.db')
    
    print("--- Financial Ratios Schema ---")
    schema_df = pd.read_sql_query("PRAGMA table_info(financial_ratios);", conn)
    print(schema_df[['name', 'type']])
    
    print("\n--- Running Screener Query ---")
    # Updated column names to match your schema exactly
    query = """
    SELECT company_id, year, return_on_equity_pct, debt_to_equity 
    FROM financial_ratios 
    WHERE return_on_equity_pct > 15 
      AND debt_to_equity < 1
    LIMIT 10;
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        if df.empty:
            print("Query executed, but returned 0 rows. (Check if your table data has been populated yet!)")
        else:
            print(df.to_string(index=False))
    except Exception as e:
        print(f"Error running query: {e}")
        
    conn.close()

if __name__ == "__main__":
    check_ratios_data()