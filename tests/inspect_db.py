import os
import duckdb
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
db_path = os.path.join(PROJECT_ROOT, "db", "analytical.duckdb")

def inspect_database():
    print(">>> Inspecting DuckDB Data Structure <<<\n")
    conn = duckdb.connect(db_path)
    
    # 1. Print the table schema
    print("1. Table Schema:")
    print(conn.execute("DESCRIBE scholarships").df()[['column_name', 'column_type']])
    print("\n" + "-"*50 + "\n")
    
    # 2. Print a sample of the specific columns causing the issue
    print("2. Sample Data for Nationality and Academic Level:")
    try:
        sample_df = conn.execute("""
            SELECT scholarship_name, eligible_nationality, academic_level 
            FROM scholarships 
            LIMIT 5
        """).df()
        pd.set_option('display.max_colwidth', None)
        print(sample_df)
    except Exception as e:
        print(f"Error fetching sample data: {e}")
        
    conn.close()

if __name__ == "__main__":
    inspect_database()