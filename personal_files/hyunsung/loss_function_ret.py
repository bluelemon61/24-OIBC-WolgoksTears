import torch

def ret_customloss(predict, y):#결과 확인용 
    # predict와 y가 batch 단위로 입력된다고 가정하고 수정
    # predict와 y는 각각 (batch_size,) 또는 (batch_size, num_features) 형태의 tensor라고 가정
    
    # e1과 e2는 배치 내 모든 샘플에 대해 계산되므로 각 배치 크기만큼의 0 텐서를 초기화
    e1 = torch.zeros_like(y)
    e2 = torch.zeros_like(y)
    accuracy = torch.zeros_like(y)  # batch 내 샘플 개수만큼의 accuracy 초기화


    #코드 설명:y(label)사이즈와 일치하는 e1,e2를 초기화한다 0으로



    e1_mask = y > 0  # y > 0인 부분 마스크
    e2_mask = y <= 0  # y <= 0인 부분 마스크
    
    #코드 설명: e1_mask와 e2_mask는 텐서인 e1과 e2의 조건

   
    e1[e1_mask] = torch.abs(y[e1_mask] - predict[e1_mask]) / (y[e1_mask] + 1e-7)
    e2[e2_mask] = torch.abs(y[e2_mask] - predict[e2_mask]) / (-y[e2_mask] + 1e-7)

     # e1과 e2에서 분모에 작은 상수를 추가하여 0으로 나누는 것을 방지하도록 1e-7을 분모에 더해준다.
    
    # 정확도 계산: y > 0일 때 predict도 > 0, y <= 0일 때 predict도 <= 0이면 accuracy = 1
    accuracy[e1_mask] = (predict[e1_mask] > 0).float()  # y > 0일 때 predict > 0이면 1
    accuracy[e2_mask] = (predict[e2_mask] <= 0).float()  # y <= 0일 때 predict <= 0이면 1
    
    # e_F 계산: e1과 e2의 weighted sum - (accuracy - 0.95)
    e_F = 0.2 * e1 + 0.8 * e2 - (accuracy - 0.95)
    
    # e_F를 제곱하여 loss 값으로 반환합니다.
    # e_F는 각 샘플에 대한 값이므로 최종 loss는 batch 차원에 대해 평균을 취합니다.
    e_F_square = e_F**2
    
    e1_non_zero_mean = e1[e1 != 0].mean() if (e1 != 0).any() else torch.tensor(0.0)
    e2_non_zero_mean = e2[e2 != 0].mean() if (e2 != 0).any() else torch.tensor(0.0)
    mean_difference = torch.abs((predict - y)).mean()

    print("accuracy는 ",accuracy.mean())
    print(f"e1 평균: {e1_non_zero_mean}")
    print(f"e2 평균: {e2_non_zero_mean}")
    print("e_F는    :",e_F)
    print("e_F의 평균은",e_F.mean())
    print("predict와 y의 평균 차이는 ",mean_difference)
    return e_F_square.mean()  # batch-wise 평균 loss 반환