import pandas as pd
import os
from pathlib import Path
import numpy as np


def detect_header_row(df, max_rows=15):
    """
    Try to detect the most likely header row by scoring each row based on:
    - Number of non-empty cells
    - Number of unique values
    - Fraction of text-like values
    Returns the index (0-based) of the detected header row.
    """
    best_score = -1
    best_row = 0
    for i in range(min(max_rows, len(df))):
        row = df.iloc[i]
        non_empty = row.notnull().sum()
        unique = len(set(row.dropna()))
        text_like = np.mean([isinstance(x, str) and len(x) > 0 for x in row])
        score = non_empty + unique + 2 * text_like  # weight text-like higher
        if score > best_score:
            best_score = score
            best_row = i
    return best_row


def interpret_sheet(df, sheet_name):
    print(f"\n--- Sheet: {sheet_name} ---")
    rows, cols = df.shape
    print(f"Dimensions: {rows} rows × {cols} columns")
    print("\nFirst 15 rows preview:")
    print(df.head(15))
    
    # Detect header row
    header_row = detect_header_row(df)
    print(f"\n[Info] Detected header row: {header_row+1} (1-based index)")
    headers = df.iloc[header_row].tolist()
    data_start = header_row + 1
    df_data = df.iloc[data_start:]
    df_data.columns = headers
    df_data = df_data.reset_index(drop=True)
    
    # Show preview with detected headers
    print("\nPreview with detected headers (first 5 rows, 10 columns):")
    print(df_data.iloc[:5, :10])
    
    # Try to interpret headers and units
    print("\nColumn interpretations:")
    for col in df_data.columns[:10]:
        header = str(col)
        sample_values = df_data[col].dropna().astype(str).tolist()[:3]
        sample_text = ', '.join(sample_values)
        interpretation = ""
        if any(unit in header.lower() for unit in ["eur", "sek", "$", "kr", "usd"]):
            interpretation += "[Currency] "
        if any(unit in header.lower() for unit in ["tco2", "ton", "kg", "g", "mwh", "kwh", "gw", "mw", "tw"]):
            interpretation += "[Unit] "
        if "date" in header.lower() or "year" in header.lower():
            interpretation += "[Time/Date] "
        if "name" in header.lower() or "desc" in header.lower():
            interpretation += "[Descriptor] "
        if not interpretation:
            if any(c.isalpha() for c in header):
                interpretation = "[General/Other] "
            else:
                interpretation = "[Unnamed/Index] "
        print(f"- {header}: {interpretation}Sample values: {sample_text}")
    print("\n--- End of sheet summary ---\n")


def analyze_excel_file(file_path):
    """
    Provide a text-based, interpretive summary of each sheet in the Excel file.
    """
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        print(f"\nExcel File: {os.path.basename(file_path)}")
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        print(f"Total Sheets: {len(sheet_names)}")
        print("\nSheet Names:")
        for i, name in enumerate(sheet_names, 1):
            print(f"  {i}. {name}")
        print("\n--- Sheet Summaries ---")
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                interpret_sheet(df, sheet_name)
            except Exception as e:
                print(f"Could not read sheet '{sheet_name}': {e}")
    except Exception as e:
        print(f"Error analyzing Excel file: {str(e)}")


def analyze_excel_file_ccs(file_path):
    """
    Provide a text-based, interpretive summary of each sheet in the Excel file, focusing on CCS-related sheets.
    """
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        print(f"\nExcel File: {os.path.basename(file_path)}")
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        print(f"Total Sheets: {len(sheet_names)}")
        print("\nSheet Names:")
        for i, name in enumerate(sheet_names, 1):
            print(f"  {i}. {name}")
        print("\n--- CCS-Relevant Sheet Summaries ---")
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                # Check if sheet is CCS-relevant
                if any(keyword in sheet_name.lower() for keyword in ["carbon capture", "cc", "ccs", "carbon capture and storage", "t&s", "transport", "storage", "co2 transport"]):
                    interpret_sheet(df, sheet_name)
            except Exception as e:
                print(f"Could not read sheet '{sheet_name}': {e}")
    except Exception as e:
        print(f"Error analyzing Excel file: {str(e)}")


if __name__ == "__main__":
    current_dir = Path.cwd()
    excel_files = list(current_dir.glob("*.xls*"))
    if not excel_files:
        print("No Excel files found in the current directory.")
    else:
        print(f"Found {len(excel_files)} Excel file(s):")
        for i, file in enumerate(excel_files, 1):
            print(f"{i}. {file}")
        for file in excel_files:
            analyze_excel_file(file) 