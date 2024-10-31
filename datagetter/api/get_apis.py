import json 
import requests # requests 관련 함수는 API 사이트 참고

class get_apis:
  def __init__(self, API_KEY, API_URL):
    self.API_KEY = API_KEY
    self.API_URL = API_URL

  def actual_weather(self, date):
    file_path = f'./json_temp_files/actual_weather/{date}.json'  # 데이터를 저장할 파일 경로 정의
    # with open(file_path, 'r') as f:


    actual_weather = requests.get(f'{self.API_URL}/data/cmpt-2024/actual-weather/{date}',
                    headers={
                        'Authorization': f'Bearer {self.API_KEY}'
                    }).json()

    with open(file_path, 'w') as f:  # file_path에 정의된 경로에 JSON 파일을 쓰기 모드('w')로 열고 데이터를 저장
      json.dump(actual_weather, f) # 데이터를 JSON 파일로 저장
  
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