# 실험 설계: 초기장 × 검증 기준

이 문서는 공개 AI 기상예측모델을 한반도 사례에 공정하게 비교하는 실험 설계를 정리합니다.
핵심은 **모델 오차와 초기장 오차를 나누어 보는 것**입니다.

## 왜 초기장과 검증 기준을 바꿔 보나

- 대부분의 공개 모델은 ERA5로 학습했습니다. 현업에서는 ERA5가 아니라 IFS·GFS·KIM 같은 현업 분석장으로 돌려야 합니다. 학습 자료와 다른 분석장으로 돌리면 분포 차이 때문에 성능이 달라질 수 있습니다.
- 검증 기준 자료(ERA5인지 IFS 분석인지)를 바꾸면 모델 순위가 달라질 수 있다는 보고가 있습니다. 순위가 기준 자료에 따라 바뀌는지는 직접 확인해야 합니다.
- 태풍은 AI 모델이 강도를 약하게 예측하는 경향이 여러 검증에서 보고되었습니다. 진행 속도 편향도 모델마다 방향이 다릅니다. 그래서 진로 오차와 강도 오차, 진행 속도 편향을 따로 봅니다.

참고: 태풍위원회 IWS-20 보고와 2025년 검증 보고, Liu et al. (2024, npj Clim. Atmos. Sci.), DeMaria et al. (2024).
세부 내용은 개발 계획 문서를 참고하세요.

## 실험 행렬

| 요인 | 수준 | kwxeval 옵션 |
|---|---|---|
| 모델 | `kwxeval models` 목록 (결정론 + 앙상블) | `run --models` |
| 초기장 | `era5`, `gfs`, `ifs`, `file`(KIM 등) | `run --ic`, `--ic-path` |
| 앙상블 구성원 수 | 앙상블 모델(`kind = ens`)만 | `run --members` |
| 검증 기준 | `era5`, `ifs` | `truth --source`, `verify --truth` |
| 태풍 기준 | IBTrACS 최적경로 (RSMC 도쿄) | `verify --best-track` |

### 사례별로 쓸 수 있는 초기장

Earth2Studio 0.20은 IFS 공개자료를 **2024-03-01 이후**만, GFS를 **2021-01-01 이후**만 받습니다.
모델 대부분이 초기시각 6시간 전 상태도 쓰므로 그 시각도 범위 안이어야 합니다. `kwxeval run`은 내려받기 전에 이것을 검사합니다.

| 사례 | era5 | gfs | ifs | file (KIM) |
|---|---|---|---|---|
| `khanun_2023` | ○ | ○ | × | ○ (내부) |
| `hinnamnor_2022` | ○ | ○ | × | ○ (내부) |
| `seoul_rain_2022` | ○ | ○ | × | ○ (내부) |
| `jeju_wind_2025` | ○ | ○ | ○ | ○ (내부) |

IFS 분석을 검증 기준으로 쓰는 것도 같은 이유로 2024-03 이후 사례(`jeju_wind_2025`)만 됩니다.

### 모델과 GPU 메모리

`kwxeval models`로 확인합니다. 메모리는 Earth2Studio 0.20 모델 설명의 값입니다.

- 40 GB: FCN, SFNO, Pangu, GraphCast(small, operational), FuXi, FengWu, AIFS, AIFS2, AIFS2 ENS
- 48 GB: AIFS ENS, Aurora, Aurora 1.5 앙상블
- 80 GB: FCN3, WeatherNext 2 Cyclones

H100 80 GB 한 장이면 모두 돌릴 수 있습니다(추정, 앙상블 구성원 수와 배치 크기에 따라 다름).

## 실행 순서 (예: 카눈)

```bash
# 1) ERA5 초기장: 재현 가능한 기준
kwxeval run --case khanun_2023 --models pangu6 graphcast_op fuxi aifs --out runs
kwxeval run --case khanun_2023 --models aifs_ens fcn3 --members 8 --out runs

# 2) GFS 초기장: 현업 조건 (파일 이름이 pangu6@gfs.nc 처럼 저장됨)
kwxeval run --case khanun_2023 --models pangu6 graphcast_op fuxi aifs --ic gfs --out runs

# 3) KIM 초기장 (내부 자료, 결과 요약만 공개)
python - <<'EOF'
import xarray as xr
from kwxeval.icfile import prepare_ic_file
ds = xr.open_mfdataset("kim_analysis_*.nc")           # 전구 위경도 격자로 후처리된 KIM 분석
prepare_ic_file(ds, "ic/kim_khanun.nc", rename={...})  # KIM 이름 -> Earth2Studio 이름
EOF
kwxeval ic-check ic/kim_khanun.nc --case khanun_2023 --variables msl u10m v10m t2m z500
kwxeval run --case khanun_2023 --models pangu6 --ic file --ic-path ic/kim_khanun.nc --out runs
#   (실행 결과 파일 이름은 pangu6@file.nc. 필요하면 pangu6@kim.nc로 바꿔 저장)

# 4) 검증
kwxeval truth  --case khanun_2023 --out runs
kwxeval verify --case khanun_2023 --runs runs --out results --persistence
```

`jeju_wind_2025`는 IFS 초기장과 IFS 검증 기준까지 쓸 수 있습니다.

```bash
kwxeval run    --case jeju_wind_2025 --models aifs pangu6 --ic ifs --out runs
kwxeval truth  --case jeju_wind_2025 --out runs --source ifs
kwxeval verify --case jeju_wind_2025 --runs runs --out results              # ERA5 기준
kwxeval verify --case jeju_wind_2025 --runs runs --out results --truth ifs  # IFS 기준 -> results/<case>/ref_ifs/
```

## KIM 초기장 파일 만들기

`kwxeval.icfile.prepare_ic_file`은 다음을 합니다.

1. 변수 이름을 Earth2Studio 이름으로 바꿉니다(`rename`).
2. 경도를 0~360°로 맞추고 0.25° 전구 격자(721 × 1440, 위도 90 → −90)로 선형 보간합니다.
3. (time, variable, lat, lon) 하나의 배열로 저장합니다. Earth2Studio `DataArrayFile`이 읽는 형식입니다.

주의할 점은 다음과 같습니다.

- **단위는 바꾸지 않습니다.** 해면기압은 Pa, 지위는 지오퍼텐셜(m² s⁻²)이어야 합니다. `check_ic_file`이 hPa나 m 단위 실수를 중앙값 범위로 잡아냅니다.
- **전구 격자만 받습니다.** 지역 격자는 거부합니다.
- **초기시각 6시간 전 상태**도 넣어야 합니다(`required_times`).
- 모델이 요구하는 변수 목록은 GPU 기계에서 `kwxeval.icfile.model_input_variables("pangu6")`로 확인합니다. 모델마다 다르며 기압면 변수가 많습니다.
- KIM 원격자(큐브구)를 위경도로 바꾸는 단계는 이 도구 밖에서 합니다(확인 필요: 사용 중인 KIM 후처리 자료 형식).

## 결과 파일

| 파일 | 내용 |
|---|---|
| `grid_scores.csv` | `base_model`, `ic`, `reference` 열 추가. 같은 모델의 초기장별 결과를 나눠 볼 수 있음 |
| `ref_ifs/grid_scores.csv` | IFS 분석 기준 검증 결과 |
| `track_summary.csv` | 진행 속도 오차 `speed_error_kmh` 추가 |
| `track_bias.csv` | 모델·예측시간 구간별 **부호 있는 평균 오차**: 진행 방향(+ 빠름), 앞섬 비율, 진행 속도, 직각 방향(+ 오른쪽), 중심기압(+ 약함), 최대풍속(− 약함) |

KoreaWxBench(`kwb`)는 이 결과를 모아 **검증 기준별 순위 변화표**와 **초기장별 RMSE 비율표**를 만듭니다.

## 공정 비교 규칙

- 같은 사례, 같은 초기시각, 같은 예측시간, 같은 격자(가장 거친 격자로 맞춤)만 비교합니다.
- 0 h 오차는 초기장 차이입니다. 예측 성능은 6 h 이후로 봅니다(`track_bias`는 0 h를 뺍니다).
- 앙상블은 구성원 수를 같게 맞춥니다. CRPS는 구성원 수에 민감하므로 공정 CRPS(`crps_fair`)를 함께 봅니다.
- 모델 가중치의 라이선스와 결과 공개 조건을 확인한 뒤 공개합니다(특히 GraphCast, WeatherNext).
