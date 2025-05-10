# 🧠 Инновационный дашборд (Django + Plotly + Excel)

Этот проект — умная система, которая умеет:

✅ Загружать Excel-файлы с данными  
📊 Показывать красивые графики  
🔮 Строить прогнозы на будущее  
🧠 Делать кластерный анализ  
📝 Создавать PDF-отчёты  

---

## 🚀 Как установить проект

> Всё работает на Windows / Linux / macOS  
> Python 3.10 или 3.11 — обязательно!

### 1. 📁 Скачай проект

Если ты смотришь на GitHub — нажми `Code` → `Download ZIP`, потом распакуй.

Или клонируй через Git:

```bash
git https://github.com/OneForces/innovation_dashboard
cd innovation-dashboard
2. 🐍 Создай виртуальное окружение
bash

python -m venv venv
3. 🟢 Активируй его
Windows:
venv\Scripts\activate
Linux/macOS:
source venv/bin/activate
4. 📦 Установи все зависимости

pip install -r requirements.txt
📄 Пример содержимого requirements.txt:
shell
Копировать
Редактировать
Django>=4.2
pandas
plotly
prophet
statsmodels
pdfkit
openpyxl
kaleido
scikit-learn
5. ⚙️ Установи PDFKit + wkhtmltopdf
Это нужно, чтобы проект умел делать PDF-отчёты.

🔽 Скачай wkhtmltopdf отсюда:

https://wkhtmltopdf.org/downloads.html

🛠 Установи и добавь bin/ в переменную среды PATH (если Windows — просто установи по умолчанию, всё заработает).

6. 📊 Подготовь базу данных

python manage.py migrate
7. ▶️ Запусти проект

python manage.py runserver
Теперь открой браузер:
📍 http://127.0.0.1:8000/

📂 Как пользоваться
1. Зайди на /upload/
Загрузи Excel-файл с регионами, годами и значениями

Поддерживается структура: Регион | 2022 | 2023 | 2024

2. Перейди по разделам:
📈 /chart/ — Графики по регионам, фильтры по году и показателю

🔢 /regression/ — Линейная регрессия

🧠 /clustering/ — Кластеры по средним значениям

📊 /forecast/ — Прогнозирование (Prophet или ARIMA)

📝 /export/pdf/ — Экспорт всех данных в PDF

📤 /export/excel/ — Экспорт в Excel

🧾 /export/custom_pdf/ — Кастомный отчёт с галочками

🧪 Примеры Excel
Пример строки:

Регион	2022	2023	2024
Москва	1.2	1.4	1.7
Санкт-Петербург	1.1	1.3	1.6

Можно загрузить несколько листов — каждый сохранится как отдельный показатель.

🧠 Возможности
Фильтрация по региону, году и типу показателя

Выбор модели прогнозирования (Prophet, ARIMA)

Настройка глубины прогноза (на сколько лет вперёд)

Дашборд с агрегированной статистикой

Кастомный экспорт PDF с галочками

🧼 Команды для разработчика

# Очистка базы (осторожно!)
python manage.py flush

# Создание администратора (если нужно)
python manage.py createsuperuser