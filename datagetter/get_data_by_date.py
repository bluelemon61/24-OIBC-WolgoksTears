import os
from dotenv import load_dotenv

from get_apis import get_apis

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

def get_data_by_date(date):
  api = get_apis(API_KEY, API_URL)

  api.actual_weather(date)
  api.weather_forecast(date)
  api.smp_da(date)
  api.smp_rt_rc(date)
  api.elec_supply(date)


if __name__ == "__main__":
  date = '2024-10-30'
  get_data_by_date(date)