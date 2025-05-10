# analytics/utils.py
import pandas as pd
from .models import RegionData

def parse_excel(file_path):
    excel_data = pd.ExcelFile(file_path)
    for sheet in excel_data.sheet_names:
        df = excel_data.parse(sheet, skiprows=1)
        indicator_name = sheet  # или взять из первой строки вручную
        for _, row in df.iterrows():
            region = str(row.iloc[0])
            for idx, year in enumerate([2022, 2023, 2024]):
                try:
                    RegionData.objects.create(
                        region=region,
                        year=year,
                        indicator_name=indicator_name,
                        value=float(row.iloc[1 + idx])
                    )
                except Exception:
                    continue
