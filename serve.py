from pathlib import Path
from bs4 import BeautifulSoup
import dateparser
from datetime import datetime
import requests
import json
import os
import pickle
import sys
from datetime import datetime, timedelta
import mlflow.pyfunc
from cachetools import cached, TTLCache

import pandas as pd
import pytz
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

pd.options.mode.chained_assignment = None  # default='warn'

MODEL_NAME = "gradient-boosting-reg-model"
MODEL_URL = f"models:/{MODEL_NAME}/latest"

CONFIG_DIR = "./runtime_data/pipeline_config"
DATA_DIR = "./runtime_data/data"

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(CONFIG_DIR).mkdir(parents=True, exist_ok=True)

HOLIDAY_LOOK_AHEAD = 5
WEATHER_LOOK_AHEAD = 3

FORECAST_DAYS = 10
YEAR_IN_THAI = 2566

@cached(cache=TTLCache(maxsize=1024, ttl=60 * 60 * 24 * 30))
def scrap_holidays():
    # Get current year
    url = f"https://calendar.kapook.com/{YEAR_IN_THAI}/holiday"

    url = requests.get(url)
    soup = BeautifulSoup(url.content, 'html.parser')

    soup = soup.find('div', {"id": "holiday_wrap"}).find_all(
        'span', {"class": "date"})
    holidays = set()
    for x in soup:
        x = x.text
        dt = dateparser.parse(x)
        dt = datetime(dt.year - 543, dt.month, dt.day)
        holidays.add(dt.strftime("%d/%m/%Y"))

    thai_holidays = {"days": list(holidays)}

    return thai_holidays


def get_api_key():
    url = requests.get("https://www.wunderground.com/weather/th/bangkok/VTBD")
    soup = BeautifulSoup(url.content, 'html.parser')

    soup = soup.find("script", {"id": "app-root-state"}).text

    start_idx = soup.find("apiKey=")
    end_idx = soup.find("&a", start_idx)

    api_key = soup[start_idx: end_idx].replace("apiKey=", "")

    return api_key

@cached(cache=TTLCache(maxsize=1024, ttl=60 * 60))
def scrap_forecast(n_day):
    api_key = get_api_key()

    forecast_data = requests.get(f"https://api.weather.com/v3/wx/forecast/hourly/{n_day}day",
                                 params={
                                     "apiKey": api_key,
                                     "geocode": "13.923,100.601",
                                     "units": "e",
                                     "language": "en-US",
                                     "format": "json"
                                 }
                                 ).json()

    data = pd.DataFrame(forecast_data)
    df = pd.DataFrame()
    df['date'] = [datetime.fromtimestamp(d, pytz.timezone(
        "Asia/Bangkok")).strftime('%Y-%m-%d') for d in data['validTimeUtc']]
    df['time'] = [datetime.fromtimestamp(d, pytz.timezone(
        "Asia/Bangkok")).strftime('%H:%M') for d in data['validTimeUtc']]
    df['datetime'] = pd.to_datetime([datetime.fromtimestamp(d, pytz.timezone(
        "Asia/Bangkok")).strftime("%Y-%m-%d %H:%M") for d in data['validTimeUtc']], utc=True)
    df['temp (F)'] = data['temperature']
    df["feel_like (F)"] = data["temperatureFeelsLike"]
    df["dew_point (F)"] = data["temperatureDewPoint"]
    df["humidity (%)"] = data["relativeHumidity"]
    df["wind"] = data["windDirectionCardinal"]
    df["wind_speed (mph)"] = data["windSpeed"]
    df["wind_gust (mph)"] = data["windGust"]
    df["pressure (in)"] = data["pressureMeanSeaLevel"]
    df["precip (in)"] = data["qpf"]
    df["condition"] = data["wxPhraseLong"]

    return df

scrap_forecast(FORECAST_DAYS)
scrap_holidays()

app = FastAPI()


@app.post("/predict")
async def predict(payload: dict):
    # load data
    model = mlflow.pyfunc.load_model(model_uri=MODEL_URL)
    mlflow.artifacts.download_artifacts(
        MODEL_URL + "/pipeline_config", dst_path="./runtime_data")
    # Load selected feature value
    with open(os.path.join(CONFIG_DIR, "selected_feature_value.json"), encoding='utf-8') as f:
        data = json.load(f)
        target_type = data['types']
        target_provinces = data['provinces']

    with open(os.path.join(CONFIG_DIR, "scalar.pkl"), 'rb') as f:
        scaler = pickle.load(f)

    df_weather = scrap_forecast(FORECAST_DAYS)
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

    holidays = scrap_holidays()['days']

    # runtime
    type, district = payload['type'], payload['district']
    timestamp = datetime.now(pytz.timezone('Asia/Bangkok'))

    query = pd.DataFrame([{
        'type': type,
        'district': district,
        'timestamp': timestamp
    }])
    query['timestamp'] = pd.to_datetime(query['timestamp'])
    query['type'].fillna("{}", inplace=True)
    query['time_of_day'] = query['timestamp'].dt.hour

    def get_types(data):
        data = data.strip()
        data = data[1:-1]  # trim "{}"
        types = data.split(",")
        return types

    # create type feature
    for t in target_type:
        query[t] = query['type'].map(lambda x: 1 if t in get_types(x) else 0)

    # map with target_provices
    query['district'] = query['district'].map(
        lambda x: x if x in target_provinces else "other")

    # create
    for province in target_provinces:
        query[province] = query['district'].apply(
            lambda x: 1 if x == province else 0)

    def is_holiday_offset(data, offset=0):
        day = data + timedelta(days=offset)
        day_text = day.strftime("%d/%m/%Y")
        if (day_text in holidays) or (day.day_name() in ['Saturday', 'Sunday']):
            return 1
        else:
            return 0

    for l in range(HOLIDAY_LOOK_AHEAD):
        query['is_holiday_' +
              str(l)] = query['timestamp'].map(lambda x: is_holiday_offset(x, l))

    df_weather_d = df_weather.drop(columns=['date', "time", "wind", "wind_speed (mph)",
                                   "wind_gust (mph)", "pressure (in)", "precip (in)", "dew_point (F)", "condition"])

    # mark only degree of 3 ex [3,9,15,21]
    mark_number = [3, 9, 15, 21]
    mark_data = df_weather_d['datetime'].map(
        lambda x: ((x.hour in mark_number) and (x.minute == 0)))
    df_weather_mark = df_weather_d[mark_data]
    df_weather_mark.sort_values("datetime", inplace=True)
    df_weather_mark.reset_index(drop=True, inplace=True)

    def nearest_ind(items, pivot):
        # assume items are sorted
        l, r = 0, len(items)-1
        while l < r:
            mid = (l+r)//2
            d = (items[mid] - pivot).total_seconds()
            if d > 0:
                r = mid
            else:
                l = mid+1
        # print(items[l], pivot)
        return l

    # parameters
    w_lookhead_k = WEATHER_LOOK_AHEAD * len(mark_number)
    weather_features = ["temp (F)", "humidity (%)"]

    dict_features = {}
    for i in range(w_lookhead_k):
        for w in weather_features:
            dict_features[str(i) + "_" + w] = []

    for target_timestamp in query['timestamp']:
        ind_target_time = nearest_ind(
            df_weather_mark['datetime'], target_timestamp) + 1
        # ind_target_time = 0
        for i in range(w_lookhead_k):
            for w in weather_features:
                if ind_target_time+i >= len(df_weather_mark):
                    dict_features[str(
                        i) + "_" + w].append(dict_features[str(i-1) + "_" + w][-1])
                else:
                    dict_features[str(
                        i) + "_" + w].append(df_weather_mark[w][ind_target_time+i])
    for k, v in dict_features.items():
        query[k] = v

    query = query.iloc[:, 3:]

    # print(query.info())

    query[:] = scaler.transform(query)

    return model.predict(query)[0]


if __name__ == "__main__":
    if "serve" in sys.argv:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)
