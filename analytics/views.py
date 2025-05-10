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


def home_view(request):
    return redirect('upload_file')

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


# 📈 График по регионам
def chart_view(request):
    region_filter = request.GET.get("region")
    queryset = RegionData.objects.all()
    if region_filter:
        queryset = queryset.filter(region__icontains=region_filter)

    df = pd.DataFrame(list(queryset.values('region', 'year', 'value')))

    if df.empty:
        chart_html = "<b>Нет данных для отображения</b>"
    else:
        fig = px.line(df, x="year", y="value", color="region", markers=True,
                      title="Динамика инновационной активности")
        chart_html = fig.to_html(full_html=False)

    return render(request, "chart.html", {"chart": chart_html})


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

def forecast_view(request):
    data = RegionData.objects.all()
    df = pd.DataFrame(list(data.values('year', 'value')))
    if df.empty or df.shape[0] < 3:
        return HttpResponse("Недостаточно данных для прогноза")

    # Prophet требует колонки ds (дата) и y (значение)
    df = df.rename(columns={'year': 'ds', 'value': 'y'})
    df['ds'] = pd.to_datetime(df['ds'], format='%Y')

    model = Prophet(yearly_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=3, freq='Y')  # прогноз на 3 года
    forecast = model.predict(future)

    fig = model.plot(forecast)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from io import BytesIO

    buf = BytesIO()
    FigureCanvasAgg(fig).print_png(buf)
    return HttpResponse(buf.getvalue(), content_type='image/png')