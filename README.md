# ai-weather-korea-eval (kwxeval)

공개 AI 기상예측모델(FourCastNet, SFNO, Pangu-Weather, GraphCast)을 **한반도 위험기상 사례**에 돌리고,
ERA5 분석장과 비교해 성능을 평가하는 도구입니다.
모델 실행은 NVIDIA [Earth2Studio](https://github.com/NVIDIA/earth2studio)를 쓰고, 초기장과 정답은 공개 ARCO ERA5에서 받습니다.

*English summary below.*

## 무엇을 평가하나

| 항목 | 지표 |
|---|---|
| 격자 검증 (msl, u10m, v10m, t2m, z500) | 면적 가중(cos 위도) RMSE, 편차, MAE, 패턴 상관계수를 예측 시간별로 계산 |
| 기준 예측 | 지속 예측(persistence)을 함께 계산해 모델의 실제 기술을 확인 |
| 확률(앙상블) 검증 | 면적 가중 CRPS·공정 CRPS, 퍼짐-오차 비. 결정론 모델은 CRPS = MAE로 함께 비교 |
| 공간 스펙트럼 | 마지막 예측시간의 파워 스펙트럼을 ERA5와 비교해 작은 규모가 얼마나 뭉개지는지 확인 |
| 태풍 진로 | 해면기압 최저점 추적. **IBTrACS 최적경로 대비** 진로 오차를 진행 방향(빠름/느림)·직각 방향(오른쪽/왼쪽)으로 분해, 중심기압·최대풍속 오차. ERA5 분석장 진로도 같은 기준으로 평가. 앙상블은 구성원별 진로, 평균 진로, 진로 퍼짐(km) |

## 사례

`cases/cases.yaml`에 들어 있습니다. 직접 추가할 수 있습니다.

| 이름 | 내용 |
|---|---|
| `khanun_2023` | 태풍 카눈(2306). 2023년 8월 10일 거제 부근 상륙 후 한반도 종단 |
| `hinnamnor_2022` | 태풍 힌남노(2211). 2022년 9월 6일 거제 부근 상륙 |
| `seoul_rain_2022` | 2022년 8월 8일 중부지방 집중호우 (대규모 장만 평가, 강수는 제외) |
| `jeju_wind_2025` | 2025년 4월 11~12일 제주 강풍 |
| `synthetic_typhoon` | GPU 없이 시험하는 합성 사례 (가짜 자료) |

### 최적경로(IBTrACS) 준비

태풍 사례는 IBTrACS 최적경로와 비교합니다. [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)에서 서태평양(WP) CSV를 받아 `data/`에 두고, `cases/cases.yaml`의 `best_track.path`를 파일 이름에 맞추세요(파일 이름은 받은 버전에 따라 다를 수 있음). 다른 위치의 파일은 `--best-track 경로`로 줄 수 있습니다.

- `agency: WMO`(기본)는 이 해역의 담당기관인 RSMC 도쿄 값입니다(10분 평균 풍속). `KMA`도 고를 수 있지만 IBTrACS v04r01의 KMA 자료는 2015~2022년만 들어 있어 2023년 카눈에는 없을 수 있습니다.
- 기관마다 풍속 평균 시간이 달라(도쿄 10분, 미국 1분) 최대풍속 오차는 같은 기관끼리만 비교하세요.
- 최적경로 파일이 없으면 경고를 내고 ERA5 진로를 기준으로 평가합니다. 이때 `first_guess`가 추적 시작점이 됩니다.

## 설치와 실행

```bash
git clone https://github.com/milianjw8-hue/ai-weather-korea-eval.git
cd ai-weather-korea-eval
pip install -e ".[run]"        # 모델 실행까지 (GPU 필요)
# pip install -e ".[dev]"     # 검증만 (CPU)
```

```bash
kwxeval run    --case khanun_2023 --models fcn pangu6 --out runs   # GPU, 인터넷
kwxeval truth  --case khanun_2023 --out runs                       # ERA5 정답 (인터넷)
kwxeval verify --case khanun_2023 --runs runs --out results --persistence
```

노트북으로 하려면 `notebooks/run_case.ipynb`를 Colab이나 GPU 서버에서 엽니다.

### GPU 없이 시험

```bash
pip install -e ".[dev]"
python scripts/make_synthetic_case.py --out runs
kwxeval verify --case synthetic_typhoon --runs runs --out results --persistence
pytest
```

### 결과 파일 (`results/<case>/`)

| 파일 | 내용 |
|---|---|
| `grid_scores.csv` | 모델·초기시각·예측시간·변수별 RMSE, 편차, MAE, 패턴 상관 |
| `rmse_vs_lead.png`, `pattern_corr_vs_lead.png` | 예측 시간별 점수 그림 |
| `track_summary.csv`, `track_summary_by_lead.csv`, `track_components_*.png` | 최적경로 대비 위치·진행 방향·직각 방향·중심기압·최대풍속 오차 |
| `crps_vs_lead.png`, `spread_skill_vs_lead.png` | 예측 시간별 CRPS, 앙상블 퍼짐-오차 비 |
| `spectra.csv`, `spectra.png` | 마지막 공통 예측시간의 파워 스펙트럼(앙상블은 구성원 0과 평균) |
| `track_*.csv`, `tracks_*.png`, `track_error_*.png` | 태풍 진로, 진로 오차. 앙상블은 `track_<모델>_members_*.csv`도 저장 |

## 주의할 점

- **GPU 메모리:** Earth2Studio는 대부분 모델에 약 40 GB를 권장합니다. 16 GB급 GPU에서는 실패할 수 있습니다.
- **모델 라이선스:** 모델 가중치마다 사용 조건이 다릅니다(예: GraphCast 가중치는 비상업 조건). 결과를 공개하기 전에 각 모델의 라이선스를 확인하세요.
- **정답 자료의 한계:** ERA5로 초기화하고 ERA5로 검증하므로 분석장 자체의 오차는 평가하지 못합니다. 지상 관측 검증은 자매 저장소 [ai-downscale-verify](https://github.com/milianjw8-hue/ai-downscale-verify)를 쓰세요.
- **해상도:** 서로 다른 격자(0.25°, 1°)를 비교할 때는 가장 거친 격자로 맞춘 뒤 점수를 냅니다.
- **앙상블 파일:** 예측 파일에 `ensemble` 차원이 있으면 앙상블로 처리합니다(Earth2Studio 앙상블 출력과 같은 이름). `kwxeval run`은 아직 결정론 실행만 하므로, 앙상블은 Earth2Studio 앙상블 워크플로로 만든 파일을 `runs/<사례>/`에 넣어 검증하세요.
- **진로 추적:** 해면기압 최저점만 따라가는 단순 방식입니다. 약한 저기압이나 상륙 후 쇠약 단계에서는 잘못 잡을 수 있습니다.
- 이 저장소에는 코드와 합성 자료만 있습니다. 실행 결과(NetCDF)는 커밋하지 않습니다.

## 방법

지표 정의는 [`docs/METHODS.md`](docs/METHODS.md)에 있습니다.

## 라이선스

이 저장소의 코드는 MIT 라이선스입니다. Earth2Studio와 각 모델 가중치는 각자의 라이선스를 따릅니다.

---

## English summary

`kwxeval` runs open AI weather models (FourCastNet, SFNO, Pangu-Weather, GraphCast-small) through
NVIDIA Earth2Studio from ARCO ERA5 initial conditions for Korean high-impact cases (typhoons
Khanun 2023 and Hinnamnor 2022, heavy rain, strong wind), and verifies them against ERA5 with
area-weighted RMSE, bias, MAE and pattern correlation by lead time, a persistence baseline,
ensemble CRPS (plain and fair) and spread-skill ratio, power spectra against the analysis,
and an MSLP-minimum typhoon tracker scored against IBTrACS best tracks with along-track /
cross-track error decomposition and central-pressure and maximum-wind errors (per member,
ensemble mean and track spread for ensembles). A synthetic case
lets the verification run without a GPU.
