import pandas as pd
import glob
import os
import re

def analyze_scholarship_data(folder_path='./data_Json'):
    print(">>> 1. Loading and Merging Data...")
    all_data = []
    file_paths = glob.glob(os.path.join(folder_path, '*_scholarships_data*.json'))
    
    if not file_paths:
        print("No JSON files found! Please check the folder path.")
        return

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        source = 'DAAD' if 'daad' in filename.lower() else 'Scholarships_com'
        
        try:
            df_temp = pd.read_json(file_path)
            df_temp['source_script'] = source
            df_temp['source_file'] = filename
            all_data.append(df_temp)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    df = pd.concat(all_data, ignore_index=True)
    print(f"Total records loaded: {len(df)}")
    print("-" * 50)

    print("\n>>> 2. Analyzing Semantic Clash in 'degree_level'")
    # Checking what degree_level actually contains based on the source
    clash_analysis = df.groupby('source_script')['degree_level'].unique()
    for source, levels in clash_analysis.items():
        print(f"- {source} 'degree_level' unique values: {levels[:5]}...")

    print("-" * 50)
    print("\n>>> 3. Detecting Brittle Extraction Risks (h5 tags indexing bug)")
    # If deadline contains '$' or funding contains dates, the array indexing failed.
    # We check the average length of deadline string. A real date is short.
    df['deadline_length'] = df['deadline'].astype(str).apply(len)
    suspicious_deadlines = df[(df['deadline_length'] > 40) & (df['source_script'] == 'Scholarships_com')]
    print(f"Found {len(suspicious_deadlines)} suspicious 'deadline' entries (too long, likely grabbed wrong HTML tag).")
    if len(suspicious_deadlines) > 0:
        print("Samples of corrupted deadlines:")
        print(suspicious_deadlines[['scholarship_name', 'deadline']].head(3).to_string())

    print("-" * 50)
    print("\n>>> 4. Quantifying Hardcoded Defaults and 'N/A' Pollution")
    # Define our known hardcoded values from the scrapers
    pollution_flags = {
        'country': ['USA', 'Germany'],
        'funding_type': ['Fully/Partially Funded (Check Details)', 'N/A'],
        'deadline': ['N/A', 'Not Specified'],
        'description': ['N/A'],
        'awards_available': ['Varies', 'N/A']
    }

    for col, bad_values in pollution_flags.items():
        if col in df.columns:
            for bad_val in bad_values:
                count = (df[col] == bad_val).sum()
                percentage = (count / len(df)) * 100
                print(f"- Column '{col}': Contains '{bad_val}' in {count} rows ({percentage:.2f}%)")

    print("-" * 50)
    print("\n>>> 5. False USA Locations (Regex Failure Analysis)")
    # Check how many of the "USA" locations are just the fallback string "USA" vs a real matched regex address
    if 'country' in df.columns:
        usa_fallback_count = (df['country'] == 'USA').sum()
        total_scholarships_com = len(df[df['source_script'] == 'Scholarships_com'])
        if total_scholarships_com > 0:
            fallback_ratio = (usa_fallback_count / total_scholarships_com) * 100
            print(f"- Scholarships.com: {fallback_ratio:.2f}% of locations failed regex and defaulted to 'USA'.")

    return df

# تشغيل التحليل
if __name__ == "__main__":
    # تأكد من وضع مسار المجلد الصحيح الذي يحتوي على ملفات الجيسون
    df_analyzed = analyze_scholarship_data('./data_Json')