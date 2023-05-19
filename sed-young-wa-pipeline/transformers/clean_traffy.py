import pandas as pd
from collections import defaultdict
if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(df, *args, **kwargs):
    df.isna().sum()

    df['type'].fillna("{}", inplace=True)
    df.dropna(subset=['district'], inplace=True)

    def get_process_hour(data):
        dt = data['last_activity'] - data['timestamp']
        total_hour = dt.days * 24 + dt.seconds//3600
        return total_hour


    df['process_hour'] = df.apply(get_process_hour, axis=1)
    df.drop(["last_activity"], axis=1, inplace=True)

    df['time_of_day'] = df['timestamp'].dt.hour

    # reset index
    df = df.reset_index(drop=True)

    return df
