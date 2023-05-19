from datetime import datetime, timedelta
if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


def get_types(data):
    data = data.strip()
    data = data[1:-1]  # trim "{}"
    types = data.split(",")
    return types

@transformer
def transform(df_weather_mark, df, holidays, selected_feature_value, *args, **kwargs):
    """
    Create Base feature
    """

    # create type feature
    for t in selected_feature_value['types']:
        df[t] = df['type'].map(lambda x: 1 if t in get_types(x) else 0)

    target_provinces = selected_feature_value['provinces']
    # map with target_provices
    df['district'] = df['district'].map(
        lambda x: x if x in target_provinces else "other")


    # create
    for province in target_provinces:
        df[province] = df['district'].apply(lambda x: 1 if x == province else 0)

    """
    Create hliday feature
    """
    def is_holiday_offset(data, offset=0):
        day = data + timedelta(days=offset)
        day_text = day.strftime("%d/%m/%Y")
        if (day_text in holidays) or (day.day_name() in ['Saturday', 'Sunday']):
            return 1
        else:
            return 0


    look_ahead = 5  # days
    for l in range(look_ahead):
        df['is_holiday_' +
            str(l)] = df['timestamp'].map(lambda x: is_holiday_offset(x, l))

    """
    Create weather feature
    """
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
    w_lookhead_k = 3*4
    weather_features = ["temp (F)", "humidity (%)"]

    dict_features = {}
    for i in range(w_lookhead_k):
        for w in weather_features:
            dict_features[str(i) + "_" + w] = []

    missing = 0
    all = 0

    for target_timestamp in df['timestamp']:
        ind_target_time = nearest_ind(
            df_weather_mark['datetime'], target_timestamp) + 1
        # ind_target_time = 0
        for i in range(w_lookhead_k):
            for w in weather_features:
                all += 1
                if ind_target_time+i >= len(df_weather_mark):
                    missing += 1
                    dict_features[str(
                        i) + "_" + w].append(dict_features[str(i-1) + "_" + w][-1])
                else:
                    dict_features[str(
                        i) + "_" + w].append(df_weather_mark[w][ind_target_time+i])
    for k, v in dict_features.items():
        df[k] = v
    print("missing weather:", missing/all)

    print(df.shape)

    return df