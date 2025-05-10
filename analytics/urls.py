from django.urls import path
from .views import (
    upload_file, chart_view, regression_view, clustering_view,
    export_excel_view, export_pdf_view, save_chart_png, home_view, forecast_view, upload_success_view, export_forecast_pdf, export_forecast_png, custom_export_pdf_view
)


urlpatterns = [
    path('', home_view, name="home"),
    path('upload/', upload_file, name="upload_file"),  
    path('chart/', chart_view, name="chart"),
    path('regression/', regression_view, name="regression"),
    path('clustering/', clustering_view, name="clustering"),
    path('export/excel/', export_excel_view, name="export_excel"),
    path('export/pdf/', export_pdf_view, name="export_pdf"),
    path('export/chart/png/', save_chart_png, name="export_chart_png"),
    path('forecast/', forecast_view, name="forecast"),
    path('upload/success/', upload_success_view, name="upload_success"),
    path('forecast/export/pdf/', export_forecast_pdf, name="forecast_export_pdf"),
    path('forecast/export/png/', export_forecast_png, name="forecast_export_png"),
    path('export/custom_pdf/', custom_export_pdf_view, name="custom_export_pdf"),
]