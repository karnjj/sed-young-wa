import pandas as pd
from datetime import datetime
if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

def concat_datetime(data) :
  date = data['date'] + " " + data['time']
  return datetime.strptime(date, "%Y-%m-%d %H:%M")

@transformer
def transform(df_weather, *args, **kwargs):
    df_weather['datetime'] = pd.to_datetime(
        df_weather.apply(concat_datetime, axis=1))
    """## Create weather information"""

    # mark only degree of 3 ex [3,9,15,21]
    mark_number = [3, 9, 15, 21]
    mark_data = df_weather['datetime'].map(
        lambda x: ((x.hour in mark_number) and (x.minute == 0)))
    df_weather_mark = df_weather[mark_data]
    df_weather_mark.sort_values("datetime", inplace=True)
    df_weather_mark.reset_index(drop=True, inplace=True)

    return df_weather_mark