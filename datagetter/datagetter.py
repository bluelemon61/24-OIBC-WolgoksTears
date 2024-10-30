import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta

if __name__ == "__main__":
    # For running as main script
    from api.get_apis import get_apis
else:
    # For using as a package
    from .api.get_apis import get_apis

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

def initialize():
  json_folder_path = './json_temp_files'
  category_folder_paths = [
    'actual_weather', 
    'weather_forecast', 
    'smp_da', 
    'smp_rt_rc', 
    'elec_supply'
  ]

  if not os.path.exists(json_folder_path):
    os.makedirs(json_folder_path)

  for folder_name in category_folder_paths:
    if not os.path.exists(f'{json_folder_path}/{folder_name}'):
      os.makedirs(f'{json_folder_path}/{folder_name}')

def get_smp_rt_rc_right_data():
  """
  폐기
  
  1시간단위 데이터를 15분단위로 수정하려고 했는데

  2024-03-01 ~ 2024-10-21 는 api로 못불러옴
  """
  target = datetime.strptime('2024-03-01', '%Y-%m-%d')
  end = datetime.strptime('2024-10-21', '%Y-%m-%d')
  
  title_row = 'ts,실시간 임시 가격(원/kWh),실시간 확정 가격(원/kWh)\n'.strip().split(',')
  content = 'ts,실시간 임시 가격(원/kWh),실시간 확정 가격(원/kWh)\n'

  api = get_apis(API_KEY, API_URL)

  while target <= end:
    target_date = target.strftime('%Y-%m-%d')
    api.smp_rt_rc(target_date)
    target += timedelta(days=1)
  
    col_name_kor_en = {
      'ts': 'ts',
      '실시간 임시 가격(원/kWh)': 'smp_rt',
      '실시간 확정 가격(원/kWh)': 'smp_rc',
    }

    with open(f'./json_temp_files/smp_rt_rc/{target_date}.json', 'r', encoding='UTF8') as json_file:
      json_data = json.load(json_file)
            
      # column 순서대로 row를 만들어나간다. 마지막 value는 ','를 제거하고 \n을 붙인다.
      for row in json_data:
        new_row = ''
        for col_key in title_row:
          new_row += str(row[col_name_kor_en[col_key]]) + ','
        new_row = new_row[:-1] + '\n'
        content += new_row

  with open(f'./data_files/제주전력시장_시장전기가격_실시간가격.csv', 'w', encoding='UTF8') as csv:
    csv.write(content)



def get_data_by_date(date):
  api = get_apis(API_KEY, API_URL)

  api.actual_weather(date)
  api.weather_forecast(date)
  api.smp_da(date)
  api.smp_rt_rc(date)
  api.elec_supply(date)

def get_data_by_start_end_date(start_date, end_date):
  """
  start_date부터 end_date까지 모든 get api를 호출하여 데이터를 json으로 저장합니다.

  Args:
    start_date (str): 시작 날짜, YYYY-MM-DD
    end_date (str): 끝 날짜, YYYY-MM-DD
  """
  target = datetime.strptime(start_date, '%Y-%m-%d')
  end = datetime.strptime(end_date, '%Y-%m-%d')

  while target <= end:
    target_date = target.strftime('%Y-%m-%d')
    get_data_by_date(target_date)
    target += timedelta(days=1)

def split_weather_data_by_location():
  """
  기상 실측 및 예측 데이터 파일을 지역별로 파일을 분리합니다.
  """
  csv_name = {
    '기상실측데이터': 'actual_weather',
    '기상예측데이터': 'weather_forecast',
  }

  for kor_csv_name in csv_name.keys():
    for num in range(1,3):
      with open(f'./data_files/{kor_csv_name}_{num}.csv', 'r+', encoding='UTF8') as actual_weather:
        data = actual_weather.readlines()
        title_row = data[0]

        content = ''
        file_name = ''

        for i, row in enumerate(data):
          if row == title_row:
            if len(content):
              with open(f'./data_files/{csv_name[kor_csv_name]}_{num}_{file_name}.csv', 'w', encoding='UTF8') as f:
                f.write(content)
              content = ''
            file_name = data[i+1].split(',')[0]
          content += row

        with open(f'./data_files/{csv_name[kor_csv_name]}_{num}_{file_name}.csv', 'w', encoding='UTF8') as f:
          f.write(content)

def make_copy_data_without_weather():
  title_name_en_kor = {
    'elec_supply': '제주전력시장_현황데이터',
    'smp_da': '제주전력시장_시장전기가격_하루전가격',
    'smp_rt_rc': '제주전력시장_시장전기가격_실시간가격',
  }

  for en_title in title_name_en_kor.keys():
    content = ''
    with open(f'./data_files/{title_name_en_kor[en_title]}.csv', 'r', encoding='UTF8') as csv:
      data = csv.readlines()
      
      # 원본 데이터를 보존하기 위해 새로운 파일에 덮어쓸거임
      for row in data:
        content += row
    
    with open(f'./data_files/{en_title}.csv', 'w', encoding='UTF8') as target_csv:
      target_csv.write(content)

def make_json_to_csv(start_date, end_date):
  """
  start_date부터 end_date까지의 json 파일들을 csv파일에 append 합니다.

  Args:
    start_date (str): 시작 날짜, YYYY-MM-DD
    end_date (str): 끝 날짜, YYYY-MM-DD
  """

  target = datetime.strptime(start_date, '%Y-%m-%d')
  end = datetime.strptime(end_date, '%Y-%m-%d')

  while target <= end:
    target_date = target.strftime('%Y-%m-%d')

    title_name_en_kor = {
      'elec_supply': '제주전력시장_현황데이터',
      'smp_da': '제주전력시장_시장전기가격_하루전가격',
      'smp_rt_rc': '제주전력시장_시장전기가격_실시간가격',
    }

    col_name_kor_en = {
      'ts': 'ts',
      '공급능력(kW)': 'supply_power',
      '현재 수요(kW)': 'present_load',
      '태양광 발전량kW)': 'power_solar',
      '풍력 발전량(kW)': 'power_wind',
      '신재생 발전량 총합(kW)': 'renewable_energy_total',
      '공급 예비력(kW)': 'supply_capacity',
      '운영 예비력(kW)': 'operation_capacity',

      '하루전가격(원/kWh)': 'smp_da',
      '실시간 임시 가격(원/kWh)': 'smp_rt',
      '실시간 확정 가격(원/kWh)': 'smp_rc',
    }

    for en_title in title_name_en_kor.keys():
      with open(f'./json_temp_files/{en_title}/{target_date}.json', 'r', encoding='UTF8') as json_file:
        json_data = json.load(json_file)

        title_row = []
        content = ''

        with open(f'./data_files/{title_name_en_kor[en_title]}.csv', 'r', encoding='UTF8') as csv:
          # json 파일의 key 순서와 csv 파일의 column 순서가 다를 수 있으므로 title_row에
          # column 순서를 저장한다.
          data = csv.readlines()
          title_row = data[0].strip().split(',')
        
        # column 순서대로 row를 만들어나간다. 마지막 value는 ','를 제거하고 \n을 붙인다.
        for row in json_data:
          new_row = ''
          for col_key in title_row:
            new_row += str(row[col_name_kor_en[col_key]]) + ','
          new_row = new_row[:-1] + '\n'
          content += new_row

        with open(f'./data_files/{en_title}.csv', 'a', encoding='UTF8') as target_csv:
          target_csv.write(content)



    # 날씨 데이터가 복잡하므로 따로 분리

    weather_type = ['actual_weather', 'weather_forecast']

    csv_name = {
      'actual_weather_1': '기상실측데이터_1',
      'actual_weather_2': '기상실측데이터_2',
      'weather_forecast_1': '기상예측데이터_1',
      'weather_forecast_2': '기상예측데이터_2',
    }

    for folder_name in weather_type:
      with open(f'./json_temp_files/{folder_name}/{target_date}.json', 'r', encoding='UTF8') as json_file:
        json_data = json.load(json_file)

        # file_name: json 파일의 key
        # [actual_weather_1, actual_weather_2], [weather_forecast_1, weather_forecast_2]
        for file_name in json_data.keys():

          # json 파일의 key 순서와 csv 파일의 column 순서가 다를 수 있으므로 title_row에
          # column 순서를 저장한다.
          title_row = []
          with open(f'./data_files/{csv_name[file_name]}.csv', 'r', encoding='UTF8') as csv:
            title_row = csv.readlines()[0].strip().split(',')

          # 지역별로 분리된 csv 파일에 각각 저장하므로
          # location과 content 초기화
          location = ''
          content = ''

          # json의 row를 순서대로 읽으며 row 한줄씩 쌓아나간다.
          for row in json_data[file_name]:
            new_row = ''

            # row 한줄씩 읽다가 location이 다른 row가 나타나면 content에 쌓아 둔 row들을
            # 해당 지역의 csv에 append하고 content 초기화 및 location 변경을 수행한다.
            if row['location'] != location:
              if len(location):
                with open(f'./data_files/{file_name}_{location}.csv', 'a', encoding='UTF8') as target_csv:
                  target_csv.write(content)
              content = ''
              location = row['location']
            
            # column 순서대로 row를 만들어나간다. 마지막 value는 ','를 제거하고 \n을 붙인다.
            for col_key in title_row:
              new_row += str(row[col_key]) + ','
            new_row = new_row[:-1] + '\n'
            content += new_row
          
          # 위 알고리즘으로 마지막 지역이 저장 되지 않으므로 마지막 지역의 데이터를 저장하는 파트
          with open(f'./data_files/{file_name}_{location}.csv', 'a', encoding='UTF8') as target_csv:
            target_csv.write(content)
    target += timedelta(days=1)

def weather_data_merger():
  location1 = [
    'Bonggae-dong',
    'Cheonji-dong', 
    'Geumak-ri', 
    'Gwangryeong-ri', 
    'Hacheon-ri',
    'Ilgwa-ri',
    'Sangmo-ri',
    'Songdang-ri',
    'Yongsu-ri',
  ]
  location2 = [
    'Cheju-do',
    'Gaigeturi',
    'Jeju',
  ]

  for weather_name in ['actual_weather', 'weather_forecast']:
    with open(f'./data_files/{weather_name}_1.csv', 'w', encoding='UTF8') as f_to:
      content = ''
      for location in location1:
        with open(f'./data_files/{weather_name}_1_{location}.csv', 'r', encoding='UTF8') as f_from:
          data = f_from.readlines()

          print(data)
          
          if len(content) == 0:
            content += data[0]

          for idx, row in enumerate(data):
            if idx > 0:
              content += row
      f_to.write(content)
    
    with open(f'./data_files/{weather_name}_2.csv', 'w', encoding='UTF8') as f_to:
      content = ''
      for location in location2:
        with open(f'./data_files/{weather_name}_2_{location}.csv', 'r', encoding='UTF8') as f_from:
          data = f_from.readlines()
          
          if len(content) == 0:
            content += data[0]

          for idx, row in enumerate(data):
            if idx > 0:
              content += row
      f_to.write(content)

def data_getter(start, end):
  initialize()
  get_data_by_start_end_date(start, end)
  split_weather_data_by_location()
  make_copy_data_without_weather()
  make_json_to_csv(start, end)
  weather_data_merger()

  print("완료되었습니다.")

if __name__ == "__main__":
  start = '2024-10-23'
  end = '2024-10-30'

  # initialize()
  # get_data_by_start_end_date(start, end)
  # split_weather_data_by_location()
  # make_copy_data_without_weather()
  # make_json_to_csv(start, end)
  weather_data_merger()

  print("완료되었습니다.")