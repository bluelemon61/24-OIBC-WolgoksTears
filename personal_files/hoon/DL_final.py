import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(128, 256, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(256, 256, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(256, 256, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(256, 256, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(256, 256, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(256, 128, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(128, 32, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 8, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(8, output_size, dtype=torch.float64)
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)

        return out
    

class ElecDataset(Dataset):

  #초기화 과정에서 데이터 전처리
  def __init__(self, x_data, y_data):
    scaler = MinMaxScaler() # 데이터의 값을 0에서 1 사이로 변환하여 모델 학습이 더 안정적으로 진행

    columns_to_scale = x_data.columns[1:] # x_data의 첫 번째 열(datetime)을 제외한 나머지 열들을 저장
    x_data[columns_to_scale] = scaler.fit_transform(x_data[columns_to_scale]) # scaler를 이용해 columns_to_scale에 해당하는 열의 값을 0에서 1 사이로 변환. 훈련 데이터므로 fit_transform 사용

  # 스케일링 후 빈 값(NaN)을 모두 0으로 저장. (누락된 값으로 인해 발생할 수 있는 오류 방지)
    self.x_data = x_data.fillna(0)
    self.y_data = y_data.fillna(0)

  # 모델 학습 시 __getitem__ 메서드를 통해 데이터셋에서 데이터를 하나씩 가져옴.
  def __getitem__(self, index):
    target_y = self.y_data['하루전가격(원/kWh)'].iloc[index] # self.y_data의 하루전가격(원/kWh)라는 열에서 index에 해당하는 데이터를 target_y 변수에 저장
    targets = self.x_data.drop(columns='datetime').iloc[index].to_numpy() # self.x_data에서 datetime 열을 제외하고 index에 해당하는 데이터를 targets 변수에 저장. to_numpy()를 사용해 PyTorch 모델에 입력할 수 있도록 변경.

  # 넘파이 배열로 변환된 targets와 target_y를 torch.tensor 형식으로 변환해 반환
    return torch.from_numpy(targets), torch.tensor(target_y)

  # 데이터셋의 전체 길이를 반환하는 메서드로, 데이터셋에 있는 총 데이터 개수
  def __len__(self):
    return int(len(self.y_data)) # self.y_data의 길이를 정수로 변환해 반환


# model: 학습할 모델 객체
# train_loader: 학습 데이터를 제공하는 데이터 로더
# criterion: 손실 함수
# optimizer: 모델의 가중치를 조정해 예측 성능을 개선하는 최적화 도구
# num_epochs: 학습을 몇 번 반복할지 정하는 값
# device: GPU나 CPU 같은 연산 장치
def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model = model.to(device)  # Move model to GPU/CPU

    train_history = [] # 각 에포크(epoch)의 손실 값을 저장하는 리스트

    # num_epochs 만큼 반복해 학습을 진행
    for epoch in range(num_epochs):
        model.train()  # 학습 모드로 설정

        running_loss = 0.0  # 손실 값 누적을 위해 0으로 초기화

        # train_loader에서 데이터를 하나씩 가져오는 반복문. tqdm은 진행 상황을 시각적으로 표시, ncols=100은 진행 막대의 너비를 설정
        for inputs, targets in tqdm(train_loader, ncols=100):

            # 모델과 동일한 장치(GPU/CPU)로 입력 데이터와 목표 값(targets)을 이동 -> 모델이 데이터에 접근 가능
            inputs, targets = inputs.to(device), targets.to(device)

            # 모델에 inputs 데이터를 입력해 예측 값을 outputs에 저장
            outputs = model(inputs)

            # criterion을 사용해 모델의 예측 값(outputs)과 실제 값(targets) 사이의 오차 계산해 loss에 저장
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()  # 이전 계산의 기울기 값 삭제 -> 이전 학습의 영향 없이 새로운 기울기를 계산
            loss.backward()        # loss를 기준으로 모델의 모든 가중치에 대한 기울기를 계산 -> 모델이 어떻게 개선되어야 할지 방향 알기
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()        # 계산된 기울기를 바탕으로 모델의 가중치를 조정하여 학습을 진행

            running_loss += loss.item() # 현재 배치(batch)에서 계산된 loss 값을 running_loss에 추가. loss.item()은 텐서 형식의 loss 값을 숫자로 변환.

            # print(loss.item())

        # Print the loss after each epoch
        avg_loss = running_loss / len(train_loader) # train_loader의 전체 배치에 대한 평균 손실을 계산 -> 현재 에포크의 전체 손실 수준을 확인
        train_history.append(avg_loss) # 계산된 평균 손실을 train_history에 저장 -> 학습이 잘 진행되는지 분석할 때 유용
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}") # 현재 에포크의 손실 출력. epoch+1은 현재 에포크 숫자, avg_loss:.4f는 평균 손실 값

    print("Training complete.")
    return model, train_history # 학습이 끝난 모델과 손실 기록이 담긴 train_history를 반환