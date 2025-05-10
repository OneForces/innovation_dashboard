import os
import io
import pandas as pd
import plotly.express as px
import plotly.io as pio
import pdfkit
from plotly.io import to_image, kaleido
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.shortcuts import redirect
from .forms import ExcelUploadForm
from .models import RegionData
from .utils import parse_excel
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from prophet.plot import plot_plotly
from statsmodels.tsa.arima.model import ARIMA

def home_view(request):
    data = RegionData.objects.all()
    df = pd.DataFrame(list(data.values('region', 'year', 'value', 'indicator_name')))

    if df.empty:
        return render(request, 'home.html', {"message": "Нет данных для отображения"})

    summary = {
        "region_count": df["region"].nunique(),
        "years": sorted(df["year"].unique()),
        "indicator_count": df["indicator_name"].nunique(),
        "max_value": df["value"].max(),
        "min_value": df["value"].min(),
        "mean_value": df["value"].mean(),
    }

    fig = px.bar(
        df.groupby("year")["value"].mean().reset_index(),
        x="year", y="value",
        title="📊 Среднее значение по годам"
    )
    chart_html = fig.to_html(full_html=False)

    return render(request, "home.html", {
        "summary": summary,
        "chart": chart_html,
    })

# 📥 Загрузка Excel
def upload_file(request):
    if request.method == 'POST':
        form = ExcelUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            file_path = form.instance.file.path
            parse_excel(file_path)
            return redirect('upload_success')
    else:
        form = ExcelUploadForm()
    return render(request, 'upload.html', {'form': form})

def upload_success_view(request):
    return render(request, 'upload_success.html')

# 📈 График по регионам
def chart_view(request):
    region_filter = request.GET.get("region")
    year_filter = request.GET.get("year")
    indicator_filter = request.GET.get("indicator")

    queryset = RegionData.objects.all()

    if region_filter:
        queryset = queryset.filter(region__icontains=region_filter)
    if year_filter:
        queryset = queryset.filter(year=year_filter)
    if indicator_filter:
        queryset = queryset.filter(indicator_name__icontains=indicator_filter)

    df = pd.DataFrame(list(queryset.values('region', 'year', 'value', 'indicator_name')))

    if df.empty:
        chart_html = "<b>Нет данных для отображения</b>"
    else:
        fig = px.line(
            df,
            x="year",
            y="value",
            color="region",
            title="Динамика инновационной активности",
            line_group="indicator_name",
            markers=True
        )
        chart_html = fig.to_html(full_html=False)

    # Вытащим уникальные значения для фильтров
    years = RegionData.objects.values_list('year', flat=True).distinct().order_by('year')
    indicators = RegionData.objects.values_list('indicator_name', flat=True).distinct()

    return render(request, "chart.html", {
        "chart": chart_html,
        "years": years,
        "indicators": indicators,
    })


# 🔢 Линейная регрессия
def regression_view(request):
    data = RegionData.objects.all()
    df = pd.DataFrame(list(data.values('year', 'value')))

    if df.empty:
        return HttpResponse("Недостаточно данных")

    X = df[['year']]
    y = df['value']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)

    fig = px.scatter(df, x='year', y='value', title=f'Линейная регрессия (R²={score:.2f})')
    fig.add_traces(px.line(x=df['year'], y=y_pred).data)

    return render(request, 'regression.html', {'chart': fig.to_html(full_html=False)})


# 🧠 Кластеризация
def clustering_view(request):
    data = RegionData.objects.all()
    df = pd.DataFrame(list(data.values('region', 'year', 'value')))

    if df.empty:
        return HttpResponse("Нет данных для кластеризации")

    grouped = df.groupby("region")["value"].mean().reset_index()
    kmeans = KMeans(n_clusters=3, random_state=42)
    grouped["cluster"] = kmeans.fit_predict(grouped[["value"]])

    fig = px.bar(grouped, x="region", y="value", color="cluster",
                 title="Кластеризация регионов по уровню инновационной активности")

    return render(request, "clustering.html", {"chart": fig.to_html(full_html=False)})


# 📤 Экспорт в Excel
def export_excel_view(request):
    data = RegionData.objects.all().values('region', 'year', 'indicator_name', 'value')
    df = pd.DataFrame(list(data))

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Инновации')
    output.seek(0)

    response = HttpResponse(
        output.read(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename="innovation_data.xlsx"'
    return response


# 🖼 Сохранение графика как PNG
def save_chart_png(request):
    data = RegionData.objects.all().values('region', 'year', 'value')
    df = pd.DataFrame(list(data))

    if df.empty:
        return HttpResponse("Нет данных")
    pio.kaleido.scope._plotlyjs = None
    fig = px.line(df, x="year", y="value", color="region", title="График инновационной активности")
    img_bytes = pio.to_image(fig, format="png")
    with open("chart.png", "wb") as f:
        f.write(img_bytes)

    with open("chart.png", "rb") as f:
        return HttpResponse(f.read(), content_type="image/png")


# 📄 Экспорт PDF с графиком
def export_pdf_view(request):
    data = RegionData.objects.all().values('region', 'year', 'indicator_name', 'value')
    df = pd.DataFrame(list(data))

    html = render_to_string("report_with_chart.html", {
        "data": df,
        "chart_path": None  # временно убираем график
    })

    try:
        pdf = pdfkit.from_string(html, False)
    except Exception as e:
        return HttpResponse(f"<h2>PDF ошибка: {e}</h2>", status=500)

    return HttpResponse(pdf, content_type='application/pdf')

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

def forecast_view(request):
    region_filter = request.GET.get("region")
    model_choice = request.GET.get("model", "prophet")
    try:
        periods = int(request.GET.get("periods") or 5)
    except ValueError:
        periods = 5

    queryset = RegionData.objects.all()
    if region_filter:
        queryset = queryset.filter(region__icontains=region_filter)

    df = pd.DataFrame(list(queryset.values('year', 'value')))

    if df.empty:
        return HttpResponse("Недостаточно данных для прогноза")

    df = df.sort_values('year')
    df['ds'] = pd.to_datetime(df['year'], format='%Y')
    df['y'] = df['value']

    regions = RegionData.objects.values_list('region', flat=True).distinct()

    # 🔮 Prophet
    if model_choice == "prophet":
        model = Prophet(yearly_seasonality=True)
        model.fit(df[['ds', 'y']])
        future = model.make_future_dataframe(periods=periods, freq='Y')
        forecast = model.predict(future)
        fig = plot_plotly(model, forecast)
        chart_html = pio.to_html(fig, full_html=False)

    # 📉 ARIMA
    elif model_choice == "arima":
        df.set_index('ds', inplace=True)
        model = ARIMA(df['y'], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=periods)
        forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), periods=periods, freq='Y')

        forecast_values = forecast.predicted_mean

        forecast_df = pd.DataFrame({
            "ds": forecast_index,
            "y": forecast_values
        })

        combined_df = pd.concat([df.reset_index(), forecast_df])
        combined_df["type"] = ["Исторические"] * len(df) + ["Прогноз"] * periods

        fig = px.line(combined_df, x="ds", y="y", color="type", markers=True,
                      title="Прогноз ARIMA")
        chart_html = fig.to_html(full_html=False)

    else:
        return HttpResponse("❌ Неизвестная модель прогноза", status=400)

    return render(request, 'forecast.html', {
        'chart': chart_html,
        'regions': regions,
        'selected_model': model_choice,
    })



def export_forecast_pdf(request):
    df = pd.DataFrame(RegionData.objects.all().values('year', 'value'))

    if df.empty:
        return HttpResponse("Нет данных для прогноза")

    df['ds'] = pd.to_datetime(df['year'], format='%Y')
    df['y'] = df['value']

    model = Prophet(yearly_seasonality=True)
    model.fit(df[['ds', 'y']])

    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)

    fig = model.plot(forecast)
    img_bytes = pio.to_image(fig, format="png")

    # Сохраняем PNG временно
    chart_path = "forecast_chart.png"
    with open(chart_path, "wb") as f:
        f.write(img_bytes)

    html = render_to_string("forecast_report.html", {
        "chart_path": os.path.abspath(chart_path),
        "data": df,
    })

    pdf = pdfkit.from_string(html, False)
    return HttpResponse(pdf, content_type="application/pdf")


def export_forecast_png(request):
    df = pd.DataFrame(RegionData.objects.all().values('year', 'value'))

    if df.empty:
        return HttpResponse("Нет данных для прогноза")

    df['ds'] = pd.to_datetime(df['year'], format='%Y')
    df['y'] = df['value']

    model = Prophet(yearly_seasonality=True)
    model.fit(df[['ds', 'y']])

    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)

    fig = model.plot(forecast)
    img_bytes = pio.to_image(fig, format="png")
    return HttpResponse(img_bytes, content_type="image/png")

def custom_export_pdf_view(request):
    include_table = request.GET.get("table") == "on"
    include_chart = request.GET.get("chart") == "on"
    include_clusters = request.GET.get("clusters") == "on"

    data = RegionData.objects.all().values('region', 'year', 'indicator_name', 'value')
    df = pd.DataFrame(list(data))

    chart_path = None
    cluster_path = None

    if include_chart:
        fig = px.line(df, x="year", y="value", color="region", title="График инновационной активности")
        img_bytes = pio.to_image(fig, format="png")
        chart_path = "chart_tmp.png"
        with open(chart_path, "wb") as f:
            f.write(img_bytes)

    if include_clusters:
        grouped = df.groupby("region")["value"].mean().reset_index()
        kmeans = KMeans(n_clusters=3, random_state=42)
        grouped["cluster"] = kmeans.fit_predict(grouped[["value"]])
        fig2 = px.bar(grouped, x="region", y="value", color="cluster", title="Кластеры по инновациям")
        img_bytes2 = pio.to_image(fig2, format="png")
        cluster_path = "cluster_tmp.png"
        with open(cluster_path, "wb") as f:
            f.write(img_bytes2)

    html = render_to_string("custom_report.html", {
        "data": df,
        "include_table": include_table,
        "chart_path": os.path.abspath(chart_path) if chart_path else None,
        "cluster_path": os.path.abspath(cluster_path) if cluster_path else None,
    })

    pdf = pdfkit.from_string(html, False)
    return HttpResponse(pdf, content_type='application/pdf')
