import urllib.request
import os
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def load_data_from_api(*args, **kwargs):
    url = 'https://publicapi.traffy.in.th/dump-csv-chadchart/bangkok_traffy.csv'

    urllib.request.urlretrieve(url, os.path.join(kwargs['DATA_DIR'], 'bangkok_traffy.csv'))

    return