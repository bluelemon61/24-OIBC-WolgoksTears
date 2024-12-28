# 2024 OIBC Challenge

- [대회 홈페이지 - postech](https://competition.postech.ac.kr/#about)
- [대회 홈페이지 - dataen](https://dataen.ai/challenge)

![Poster](./Main_Poster.png)

# 대회 내용

> 과거 전기가격 및 전력 시장 현황 데이터를 이용하여 대회 5일간 내일의 전기가격을 예측

# 기여 내용

- `대회의 산술 공식에 맞는 Custom Loss 제작`
  - 대회 산술공식 + α × Zero Penalty
  - 마이너스 패널티: 대회 산술공식으로 인해 전기가격이 0에 가깝게 형성 될 시 오차 패널티가 크게 작용하여 이를 상쇄하고자 Loss를 극대화 시킴
  - α: 조절가능한 상수
- `빠른 제출을 위한 API 훅`
- `Pandas를 이용한 데이터 전처리`

# Contributors

- [이동훈](https://github.com/bluelemon61)
- [조현성](https://github.com/hyunsung1221)
- [곽도연](https://github.com/Karryun)
