import pandas as pd
import os
from typing import List, Dict

class SpreadsheetProcessor:
    def process_spreadsheet(self, filepath: str) -> List[Dict]:
        """
        Processes a spreadsheet file (CSV, XLSX, XLS) and extracts its content.
        Returns a list of dictionaries, where each dictionary represents a row
        or a sheet's content.
        """
        file_extension = filepath.rsplit('.', 1)[1].lower()
        extracted_data = []
        file_name = os.path.basename(filepath)

        try:
            if file_extension == 'csv':
                df = pd.read_csv(filepath)
                for i, row in df.iterrows():
                    extracted_data.append({
                        'file_path': file_name,
                        'page_number': f"row_{i+1}", # Differentiate rows
                        'full_text': row.to_string()
                    })
            elif file_extension in {'xlsx', 'xls'}:
                xls = pd.ExcelFile(filepath)
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    for i, row in df.iterrows():
                        extracted_data.append({
                            'file_path': file_name,
                            'page_number': f"{sheet_name}_row_{i+1}", # Differentiate rows
                            'full_text': row.to_string()
                        })
            else:
                print(f"Unsupported spreadsheet format: {file_extension}")
                return []
        except Exception as e:
            print(f"Error processing spreadsheet {filepath}: {e}")
            return []
        
        return extracted_data
