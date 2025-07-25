# nasdaq_premarket_ai.py (Polygon версия)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
import warnings

warnings.filterwarnings("ignore")

API_KEY = "2SrZuWhnRJO5me29ApXRwF9dumhy2PYw"
SYMBOL = "QQQ"
DAYS_BACK = 7

berlin_tz = pytz.timezone("Europe/Berlin")

def get_intraday_data(symbol, date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{date}/{date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 1000,
        "apiKey": API_KEY
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(f"⚠️ Ошибка загрузки {date}: {r.text}")
        return pd.DataFrame()
    raw = r.json().get("results", [])
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_convert("Europe/Berlin")
    df.set_index('timestamp', inplace=True)
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def extract_features_and_labels():
    result = []
    skipped = []
    for i in range(DAYS_BACK):
        date = (datetime.now(tz=berlin_tz) - timedelta(days=i)).strftime("%Y-%m-%d")
        df = get_intraday_data(SYMBOL, date)
        if df.empty:
            skipped.append((date, 0, 0))
            continue
        df['Date'] = df.index.date
        morning = df.between_time("10:00", "11:00")
        full_pre = df.between_time("13:00", "13:30")
        if len(morning) < 10 or len(full_pre) < 10:
            skipped.append((date, len(morning), len(full_pre)))
            continue
        open_morning = morning['Open'].iloc[0]
        close_morning = morning['Close'].iloc[-1]
        change_morning = (close_morning - open_morning) / open_morning
        volume_morning = morning['Volume'].mean()
        open_full = full_pre['Open'].iloc[0]
        close_full = full_pre['Close'].iloc[-1]
        change_full = (close_full - open_full) / open_full
        label = "up" if change_full > 0.003 else "down" if change_full < -0.003 else "flat"
        result.append({
            "change_pre": change_morning,
            "volume_pre": volume_morning,
            "label": label
        })
    if skipped:
        print("ℹ️ Пропущено дней из-за нехватки данных:")
        for d, m, f in skipped:
            print(f" - {d}: 10–11 ({m} мин), 13–13:30 ({f} мин)")
    return pd.DataFrame(result)

data = extract_features_and_labels()

if data.empty:
    print("⚠️ Недостаточно данных для обучения модели. Прогноз не рассчитан.")
    exit(0)

X = data[["change_pre", "volume_pre"]]
y = data["label"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ========== Прогноз на сегодня ==========
today = datetime.now(tz=berlin_tz).strftime("%Y-%m-%d")
today_df = get_intraday_data(SYMBOL, today)

if today_df.empty:
    print("⏳ Данные за сегодня недоступны. Попробуйте позже.")
else:
    morning_now = today_df.between_time("10:00", "11:00")
    full_now = today_df.between_time("13:00", "13:30")

    if len(morning_now) < 5:
        print("⏳ Недостаточно данных premarket на сегодня (10:00–11:00).")
    else:
        open_pre = morning_now['Open'].iloc[0]
        close_pre = morning_now['Close'].iloc[-1]
        change_pre = (close_pre - open_pre) / open_pre
        volume_pre = morning_now['Volume'].mean()

        pred = model.predict([[change_pre, volume_pre]])[0]
        proba = model.predict_proba([[change_pre, volume_pre]])[0]
        confidence = round(max(proba) * 100, 1)

        if len(full_now) > 5:
            open_real = full_now['Open'].iloc[0]
            close_real = full_now['Close'].iloc[-1]
            change_real = (close_real - open_real) / open_real
            actual = "up" if change_real > 0.003 else "down" if change_real < -0.003 else "flat"
            print(f"✅ Факт: {actual.upper()} ({change_real*100:.2f}%)")
        else:
            actual = None

        print(f"📊 Утреннее изменение (10:00–11:00): {change_pre*100:.2f}%")
        print(f"📦 Средний объём: {volume_pre:.0f}")
        print(f"🔮 Прогноз от ИИ: {pred.upper()} (уверенность {confidence}%)")
