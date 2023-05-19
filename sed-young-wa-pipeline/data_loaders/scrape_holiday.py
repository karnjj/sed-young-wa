import requests
import os
import dateparser
from bs4 import BeautifulSoup
from datetime import datetime
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def load_data(*args, **kwargs):
    # Get current year
    url = f"https://calendar.kapook.com/{kwargs['YEAR_IN_THAI']}/holiday"

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
    
    return list(holidays)