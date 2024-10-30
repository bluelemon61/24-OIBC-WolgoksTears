import json 
import requests

class post_apis:
  def __init__(self, API_KEY, API_URL):
    self.API_KEY = API_KEY
    self.API_URL = API_URL

  def submissions(self, result_data):
    """
    결과 데이터를 제출하는 api

    Args:
      result_data (float[ ]):
      
        float 24개 (1시 ~ 24시) 배열

        예시 -> [107.39, 107.39, 95.39, 87.89, 0, ... 24개]
    """
    result = {
      'submit_result' : result_data
    }

    success = requests.post(f'{self.API_URL}/submissions/cmpt-2024',
                        data=json.dumps(result),
                        headers={
                            'Authorization': f'Bearer {self.API_KEY}'
                        }).json()

    print(success) 