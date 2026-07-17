import pandas as pd
import glob
import os

def load_all_data(folder_path='./data_Json'):
    all_data = []
    file_paths = glob.glob(os.path.join(folder_path, '*_scholarships_data*.json'))
    for file_path in file_paths:
        try:
            df_temp = pd.read_json(file_path)
            df_temp['source_file'] = os.path.basename(file_path)
            all_data.append(df_temp)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    if not all_data:
        raise ValueError("No JSON files found in the specified directory.")
    return pd.concat(all_data, ignore_index=True)

def analyze_data_to_html(df):
    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>Comprehensive Data Profiling Report</title>",
        "<style>",
        "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #2c3e50; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #ecf0f1; }",
        "h1 { color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 30px; }",
        "h2 { color: #2980b9; border-bottom: 2px solid #bdc3c7; padding-bottom: 5px; }",
        ".card { background: #fff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 25px; margin-bottom: 25px; }",
        "table { width: 100%; border-collapse: collapse; margin-top: 15px; }",
        "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }",
        "th { background-color: #f8f9fa; font-weight: 600; color: #34495e; }",
        "tr:hover { background-color: #f5f6fa; }",
        ".warning { color: #e74c3c; font-weight: bold; }",
        ".highlight { background-color: #f1c40f; padding: 3px 6px; border-radius: 4px; color: #000; }",
        "blockquote { background: #f9f9f9; border-left: 5px solid #3498db; padding: 15px; font-style: italic; margin: 15px 0; }",
        "ul { list-style-type: none; padding: 0; }",
        "li { margin-bottom: 8px; padding: 8px; background: #f8f9fa; border-radius: 4px; }",
        ".metric { font-size: 1.1em; color: #7f8c8d; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1> Comprehensive Data Profiling Report</h1>",
        f"<div class='card'><h2>Overview</h2><p class='metric'><strong>Total Records:</strong> {len(df):,}</p><p class='metric'><strong>Total Features (Columns):</strong> {len(df.columns)}</p></div>"
    ]

    # 1. DATA QUALITY
    html.append("<div class='card'><h2>1. Data Quality & Completeness</h2>")
    html.append("<table><tr><th>Column Name</th><th>Missing / Empty / N/A</th><th>Missing Percentage</th><th>Status</th></tr>")
    
    missing_data = df.isnull().sum()
    empty_strings = (df == "").sum()
    na_strings = (df == "N/A").sum() + (df == "Not Specified").sum()

    for col in df.columns:
        total_missing = missing_data[col] + empty_strings.get(col, 0) + na_strings.get(col, 0)
        percentage = (total_missing / len(df)) * 100
        
        status = "Healthy"
        color_style = "color: #27ae60; font-weight: bold;"
        
        if percentage > 50:
            status = "Critical (Drop/Impute)"
            color_style = "color: #e74c3c; font-weight: bold;"
        elif percentage > 20:
            status = "Warning (Needs attention)"
            color_style = "color: #f39c12; font-weight: bold;"
        
        html.append(f"<tr><td><strong>{col}</strong></td><td>{total_missing:,}</td><td>{percentage:.2f}%</td><td style='{color_style}'>{status}</td></tr>")
    html.append("</table>")

    exact_duplicates = df.duplicated(subset=['scholarship_name', 'application_link']).sum()
    html.append(f"<p style='margin-top:20px; font-size: 1.1em;'><strong>Exact Duplicates (Name + Link):</strong> <span class='highlight'>{exact_duplicates} rows</span></p>")
    html.append("</div>")

    # 2. CATEGORICAL
    html.append("<div class='card'><h2>2. Categorical Distribution (Top Values)</h2>")
    categorical_cols = ['country', 'funding_type', 'degree_level', 'source_file']
    for col in categorical_cols:
        if col in df.columns:
            html.append(f"<h3>Variable: {col} <span style='font-size:0.8em; color:#7f8c8d;'>(Unique Values: {df[col].nunique()})</span></h3><ul>")
            top_values = df[col].value_counts().head()
            for val, count in top_values.items():
                html.append(f"<li><strong>{val}</strong>: {count:,} ({(count/len(df))*100:.1f}%)</li>")
            html.append("</ul>")
    html.append("</div>")

    # 3. TEXTUAL
    html.append("<div class='card'><h2>3. Textual Analysis (RAG Chunking Metrics)</h2>")
    text_cols = ['description', 'eligibility', 'scholarship_details', 'application_process']
    for col in text_cols:
        if col in df.columns:
            words_count = df[col].astype(str).apply(lambda x: len(x.split()) if x not in ['N/A', 'None', ''] else 0)
            valid_texts = words_count[words_count > 0]
            if not valid_texts.empty:
                max_w = valid_texts.max()
                max_html = f"<span class='warning'>{max_w}</span> (Warning: Token overflow risk)" if max_w > 300 else str(max_w)
                
                html.append(f"<h3>Field: {col}</h3><ul>")
                html.append(f"<li><strong>Min Words:</strong> {valid_texts.min()}</li>")
                html.append(f"<li><strong>Max Words:</strong> {max_html}</li>")
                html.append(f"<li><strong>Average Words:</strong> {valid_texts.mean():.1f}</li>")
                
                dead_texts = (valid_texts < 10).sum()
                dead_html = f"<span class='warning'>{dead_texts}</span>" if dead_texts > 0 else "0"
                html.append(f"<li><strong>Dead/Short Texts (< 10 words):</strong> {dead_html} rows</li>")
                html.append("</ul>")
    html.append("</div>")

    # 4. BOILERPLATE
    html.append("<div class='card'><h2>4. Boilerplate Detection (Text Duplication)</h2>")
    if 'eligibility' in df.columns:
        eligibility_counts = df[df['eligibility'] != 'N/A']['eligibility'].value_counts()
        boilerplate = eligibility_counts[eligibility_counts > 5]
        
        if not boilerplate.empty:
            html.append("<p class='warning'>⚠ Warning: Found exact matching paragraphs across multiple scholarships. This ruins Semantic Search.</p>")
            html.append(f"<blockquote>{str(boilerplate.index[0])[:300]}...</blockquote>")
            html.append(f"<p><strong>Repeated exactly:</strong> <span class='highlight'>{boilerplate.iloc[0]} times</span>.</p>")
        else:
            html.append("<p>✅ No major boilerplate text detected in eligibility criteria.</p>")
    html.append("</div>")

    html.append("</body></html>")
    return "\n".join(html)

if __name__ == "__main__":
    print(">>> Loading data...")
    try:
        df = load_all_data('./data_Json')
        print(">>> Generating HTML Report...")
        html_content = analyze_data_to_html(df)
        
        output_file = 'comprehensive_analysis_report.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"\n✅ SUCCESS! HTML Report Generated: '{output_file}'")
        print("-> Double click the file to open it in your web browser.")
    except Exception as e:
        print(f"\n❌ Error: {e}")