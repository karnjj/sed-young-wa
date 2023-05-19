import pandas as pd
import os
from datetime import datetime, timedelta

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(data, *args, **kwargs):
    path = os.path.join(kwargs['DATA_DIR'], "bangkok_traffy.csv")

    df_raw = pd.read_csv(path, sep=',', quotechar="\"", encoding='utf-8')

    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'].str.slice(stop=-3))
    df_raw['last_activity'] = pd.to_datetime(
        df_raw['last_activity'].str.slice(stop=-3))
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    df_raw['last_activity'] = pd.to_datetime(df_raw['last_activity'])

    current_day = datetime.now()
    # current_day = datetime(2023,5,12)
    end_date_train = current_day - timedelta(days=kwargs['TIME_OFFSET'])
    df = df_raw[df_raw['timestamp'] <= end_date_train]

    target_columns = ["ticket_id", "type",
                  "district", "timestamp", "last_activity"]
    df = df[~df['ticket_id'].isna()]
    df = df[df['state'] == "เสร็จสิ้น"]
    df = df[target_columns]

    return df
        