import pandas as pd
import os
import json
from collections import defaultdict
if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def transform_custom(df, *args, **kwargs):
    selected_feature_value = {"types": [], "provinces": []}
    
    """### Type
    - Filter with the frequency of each category.
    """

    # @title Fillter high cardinality

    type_freq = defaultdict(int)
    get_by = "rank"  # @param ["ratio", "rank"]
    k = 7  # @param
    ratio_value = 0.5  # @param


    def get_types(data):
        data = data.strip()
        data = data[1:-1]  # trim "{}"
        types = data.split(",")
        return types

    # get set of type
    for i in range(len(df)):
        data = get_types(df['type'][i])
        for d in data:
            type_freq[d] += 1
    type_freq.pop("")
    type_freq = sorted([(v/df.shape[0], k)
                    for k, v in type_freq.items()], reverse=True)
    type_freq

    if get_by == "ratio":
        target_type = []
        for x in type_freq:
            if x[0] > ratio_value:
                target_type.append(x[1])
    elif get_by == "rank":
        target_type = [x[1] for x in type_freq[:k]]


    selected_feature_value['types'] = target_type

    """### Province
    - Filter with the frequency of each category.
    """

    # @title Fillter high cardinality

    get_by = "rank"  # @param ["ratio", "rank"]
    k = 25  # @param
    ratio_value = 0.5  # @param

    district_freq = df.groupby("district")["district"].count()/df.shape[0]
    district_freq = sorted([(x[1], x[0])
                        for x in district_freq.items()], reverse=True)
    district_freq

    if get_by == "ratio":
        target_provinces = []
        for x in district_freq:
            if x[0] > ratio_value:
                target_provinces.append(x[1])
    elif get_by == "rank":
        target_provinces = [x[1] for x in district_freq[:k]]

    if "other" not in target_provinces:
        target_provinces += ["other"]

    selected_feature_value['provinces'] = target_provinces

    with open(os.path.join(kwargs['CONFIG_DIR'],"selected_feature_value.json"), "w") as outfile:
        json_object = json.dumps(selected_feature_value, indent = 4, ensure_ascii=False) 
        outfile.write(json_object)
    
    return selected_feature_value