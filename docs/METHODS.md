# 방법 (Methods)

## 1. 예측 생성

- Earth2Studio `deterministic` 워크플로로 각 모델을 사례의 초기시각부터 정해진 시간까지 적분합니다.
- 초기장: ARCO ERA5 (Google Cloud 공개 자료, 0.25°, 1시간 간격)
- 출력: 평가 변수(msl, u10m, v10m, t2m, z500)만 남기고 사례 영역으로 잘라 저장합니다.
- 모델 시간 간격: FCN·SFNO·Pangu6·GraphCast-small 6시간, Pangu24 24시간

## 2. 격자 검증

1. 예측과 정답을 같은 격자로 맞춥니다. 입력 중 가장 거친 해상도로 영역 안에 규칙 격자를 만들고 쌍선형 내삽합니다.
2. 유효시각(초기시각 + 예측시간)이 같은 정답장과 비교합니다.
3. 면적 가중치 $w_j = \cos\phi_j / \overline{\cos\phi}$를 씁니다.

| 지표 | 정의 |
|---|---|
| RMSE | $\sqrt{\sum w (f-o)^2 / \sum w}$ |
| 편차 | $\sum w (f-o) / \sum w$ |
| MAE | $\sum w \lvert f-o \rvert / \sum w$ |
| 패턴 상관 | 영역 가중평균을 뺀 편차장 $f'$, $o'$의 가중 상관 $\sum w f'o' / \sqrt{\sum w f'^2 \sum w o'^2}$ |

패턴 상관은 기후값을 빼지 않은 공간 상관입니다. 기후값 편차를 쓰는 이상상관계수(ACC)와는 다릅니다.

### 지속 예측

초기시각의 정답장을 모든 예측시간에 그대로 둔 예측입니다. 모델 RMSE가 지속 예측보다 작아야 의미 있는 기술이 있다고 봅니다.

## 3. 태풍 진로

- 첫 시각은 `first_guess`에서 `search_km`(기본 400~500 km) 안의 해면기압 최저 격자점을 중심으로 잡습니다.
- 다음 시각은 직전 중심에서 `search_km` 안의 최저점을 찾습니다.
- 중심 300 km 안의 10 m 최대 풍속을 강도 지표로 함께 기록합니다.
- 진로 오차는 같은 유효시각의 ERA5 중심과의 대권거리(km)입니다.
- 격자점 위치만 쓰므로 위치 분해능은 격자 간격(0.25°에서 약 28 km)으로 제한됩니다.

ERA5 진로도 같은 방법으로 추적한 값이므로 최적경로와 다를 수 있습니다. 보고할 때는 IBTrACS 또는 기상청 최적경로와 함께 제시하는 것을 권합니다.

## 참고 문헌

- Pathak, J., et al. (2022). FourCastNet: A global data-driven high-resolution weather model using adaptive Fourier neural operators. arXiv:2202.11214.
- Bonev, B., et al. (2023). Spherical Fourier neural operators: Learning stable dynamics on the sphere. arXiv:2306.03838.
- Bi, K., et al. (2023). Accurate medium-range global weather forecasting with 3D neural networks. *Nature*, 619, 533–538.
- Lam, R., et al. (2023). Learning skillful medium-range global weather forecasting. *Science*, 382, 1416–1421.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Q. J. R. Meteorol. Soc.*, 146, 1999–2049.
