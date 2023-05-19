import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pytz
import time as os_time
import random

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

def get_api_key():
  url = requests.get("https://www.wunderground.com/weather/th/bangkok/VTBD")
  soup = BeautifulSoup(url.content, 'html.parser')

  soup = soup.find("script", {"id":"app-root-state"}).text

  start_idx = soup.find("apiKey=")
  end_idx = soup.find("&a", start_idx)

  api_key = soup[start_idx: end_idx].replace("apiKey=", "")

  return api_key

def fill_missing(df):
  if len(df) != 48:
    periods = ['00:00', '00:30', '01:00', '01:30', '02:00', '02:30',
                '03:00', '03:30', '04:00', '04:30', '05:00', '05:30',
                '06:00', '06:30', '07:00', '07:30', '08:00', '08:30',
                '09:00', '09:30', '10:00', '10:30', '11:00', '11:30',
                '12:00', '12:30', '13:00', '13:30', '14:00', '14:30',
                '15:00', '15:30', '16:00', '16:30', '17:00', '17:30',
                '18:00', '18:30', '19:00', '19:30', '20:00', '20:30',
                '21:00', '21:30', '22:00', '22:30', '23:00', '23:30']
    for period in periods:
        if period not in df['time'].values:
            df = df.append({'time': period}, ignore_index=True)
    df = df.sort_values(by=['time'])
    df = df.fillna(method='ffill')
    df = df.drop_duplicates(subset=['time'], keep='first')
    df = df.reset_index(drop=True)
  return df

def scrap_weather_v2(n_day):

  api_key = get_api_key()

  all_df = pd.DataFrame()

  today = datetime.now()
  for i in range(1, n_day + 1):
    df = pd.DataFrame(columns=["date","time","temp (F)","feel_like (F)","dew_point (F)","humidity (%)","wind","wind_speed (mph)","wind_gust (mph)","pressure (in)","precip (in)","condition"])
    target_date = today - timedelta(days=i)
    historical_data = requests.get("https://api.weather.com/v1/location/VTBD:9:TH/observations/historical.json",
                                  params={
                                      "apiKey": api_key,
                                      "units": "e",
                                      "startDate": target_date.strftime("%Y%m%d")
                                  }
                                  ).json()                
    for data in historical_data['observations']:
      date_time = datetime.fromtimestamp(data['valid_time_gmt'], pytz.timezone("Asia/Bangkok"))
      date = date_time.strftime('%Y-%m-%d')
      time = date_time.strftime('%H:%M')
      new_row = {
          "date": date,
          "time": time,
          "temp (F)": data["temp"],
          "feel_like (F)": data["feels_like"],
          "dew_point (F)": data["dewPt"],
          "humidity (%)": data["rh"],
          "wind": data["wdir_cardinal"],
          "wind_speed (mph)": data["wspd"] or 0,
          "wind_gust (mph)": data["gust"] or 0,
          "pressure (in)": data["pressure"],
          "precip (in)": data["precip_hrly"] or 0,
          "condition": data["wx_phrase"],
      }

      df = df.append(new_row, ignore_index=True)

    df = df.dropna()
    df = fill_missing(df)

    all_df = all_df.append(df, ignore_index=True)

    os_time.sleep(random.random() + 0.3)

  return all_df

@data_loader
def load_data_from_api(*args, **kwargs):

    return scrap_weather_v2(n_day=kwargs['DAY_OF_HIST_WEATHER'])