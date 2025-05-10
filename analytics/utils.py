# analytics/utils.py
import pandas as pd
from .models import RegionData

def parse_excel(file_path):
    excel_data = pd.ExcelFile(file_path)
    for sheet in excel_data.sheet_names:
        df = excel_data.parse(sheet, skiprows=1)  # Пропускаем заголовок
        for _, row in df.iterrows():
            region = str(row.iloc[0])
            for year, value in zip([2022, 2023, 2024], row.iloc[1:4]):
                try:
                    RegionData.objects.create(
                        region=region,
                        year=year,
                        indicator_name="Стоимость инновационных товаров",
                        value=float(value)
                    )
                except Exception:
                    continue
