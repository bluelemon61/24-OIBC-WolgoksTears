import json 
import requests

class get_apis:
  def __init__(self, API_KEY, API_URL):
    self.API_KEY = API_KEY
    self.API_URL = API_URL

  def actual_weather(self, date):
    actual_weather = requests.get(f'{self.API_URL}/data/cmpt-2024/actual-weather/{date}',
                    headers={
                        'Authorization': f'Bearer {self.API_KEY}'
                    }).json()

    file_path = f'./json_temp_files/actual_weather/{date}.json'

    with open(file_path, 'w') as f:
      json.dump(actual_weather, f)
  
  def weather_forecast(self, date):
    weather_forecast = requests.get(f'{self.API_URL}/data/cmpt-2024/weather-forecast/{date}',
                    headers={
                        'Authorization': f'Bearer {self.API_KEY}'
                    }).json()

    file_path = f'./json_temp_files/weather_forecast/{date}.json'

    with open(file_path, 'w') as f:
      json.dump(weather_forecast, f)
  
  def smp_da(self, date):
    smp_da = requests.get(f'{self.API_URL}/data/cmpt-2024/smp-da/{date}',
                    headers={
                        'Authorization': f'Bearer {self.API_KEY}'
                    }).json()

    file_path = f'./json_temp_files/smp_da/{date}.json'

    with open(file_path, 'w') as f:
      json.dump(smp_da, f)

  def smp_rt_rc(self, date):
    smp_rt_rc = requests.get(f'{self.API_URL}/data/cmpt-2024/smp-rt-rc/{date}',
                    headers={
                        'Authorization': f'Bearer {self.API_KEY}'
                    }).json()

    file_path = f'./json_temp_files/smp_rt_rc/{date}.json'

    with open(file_path, 'w') as f:
      json.dump(smp_rt_rc, f)
  
  def elec_supply(self, date):
    elec_supply = requests.get(f'{self.API_URL}/data/cmpt-2024/elec-supply/{date}',
                    headers={
                        'Authorization': f'Bearer {self.API_KEY}'
                    }).json()

    file_path = f'./json_temp_files/elec_supply/{date}.json'

    with open(file_path, 'w') as f:
      json.dump(elec_supply, f)