# 🧬 HM-Recsys-llm-test-toy_수정중


> H&M 실 거래 데이터 모델 기반의 추천 시스템 & A/B 테스트 LLM 시뮬레이션 파이프라인

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [기술 스택](#기술-스택)
- [디렉토리 구조](#디렉토리-구조)
- [설치 및 환경 설정](#설치-및-환경-설정)
- [파이프라인 실행 가이드](#파이프라인-실행-가이드)
- [Scripts 상세 설명](#scripts-상세-설명)
- [핵심 모듈 (src)](#핵심-모듈-src)

---

## 프로젝트 개요

**HM-Recsys-llm-test-toy**는 H&M의 실제 거래 데이터(`transactions_train.csv`, `articles.csv`, `customers.csv`)를 활용하여 **개인화된 상품 추천**을 제공하는 모델을 만들고 고객 세그먼트 기반으로 페르소나를 만들어 LLM 시뮬레이션을 수행합니다.

### 핵심 기능

| 기능 | 설명 |
|---|---|
| **Feature Engineering** | DuckDB 기반 User/Item Feature 생성 (구매 빈도, 평균 가격, 최근성 등) |
| **Ranking Model** | LightGBM LambdaRank 기반 개인화 랭킹 모델 |
| **Candidate Generation** | 인기도 + 협업 필터링 + 카테고리 기반 후보군 생성 |
| **User Segmentation** | 구매 패턴 기반 유저 세그먼트 분류 (VIP / 고빈도 / 중빈도 / 저빈도) |
| **A/B Test Simulation** | 가상 유저 시뮬레이션을 통한 추천 전략 비교 (인기 상품 vs ML 추천) |
| **LLM Integration** | Ollama(로컬 LLM)를 활용한 페르소나 기반 클릭 행동 시뮬레이션 |
| **Hyperparameter Tuning** | Optuna + LightGBM 자동 하이퍼파라미터 튜닝 |

---

## 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                        Raw Data (CSV)                           │
│         transactions_train.csv / articles.csv / customers.csv   │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│               Phase 1: Data Processing                          │
│   scripts/process_data.py → data/features/*.parquet             │
│   (UserFeatureGenerator + ItemFeatureGenerator via DuckDB)      │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│               Phase 2: Model Training                           │
│   scripts/train_model.py → models/artifacts/purchase_ranker.pkl │
│   (LightGBM LambdaRank + NDCG@10,20 평가)                       │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│               Phase 3: Serving & Simulation                     │
│   scripts/run_simulation.py                                     │
│   (RecommendationService + VirtualUser + A/B Test)              │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│               Phase 4: Analysis & Optimization                  │
│   scripts/calculate_sample_size.py                              │
│   scripts/analyze_purchase_cycle.py                             │
│   (통계적 검정, 구매 주기 분석, 쇼핑 패턴 추출)                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 분류 | 기술 |
|---|---|
| **데이터 처리** | DuckDB, Polars, Pandas |
| **ML 모델** | LightGBM (LambdaRank) |
| **하이퍼파라미터 튜닝** | Optuna (`optuna.integration.lightgbm`) |
| **LLM** | Ollama (로컬 LLM 서버) |
| **통계 분석** | SciPy, NumPy |
| **언어** | Python 3.10+ |

---

## 디렉토리 구조

```
Local_Helix_01/
├── data/                          # 원본 데이터 & 생성된 Feature
│   ├── transactions_train.csv     # H&M 거래 데이터
│   ├── articles.csv               # 상품 메타데이터
│   ├── customers.csv              # 고객 메타데이터
│   └── features/                  # 생성된 Feature (Parquet)
│       ├── user_features.parquet
│       └── item_features.parquet
│
├── src/                           # 핵심 소스 코드
│   ├── data/                      # Feature Engineering 모듈
│   │   ├── user_features.py       # 유저 특성 생성기
│   │   ├── item_features.py       # 아이템 특성 생성기
│   │   └── feature_store.py       # Feature Store 관리
│   │
│   ├── models/                    # ML 모델 모듈
│   │   ├── dataset.py             # 학습 데이터셋 생성
│   │   ├── ranker.py              # LightGBM Ranker 래퍼
│   │   ├── evaluation.py          # 랭킹 메트릭 평가 (NDCG, MRR, HR)
│   │   ├── candidate_generation.py # 후보군 생성 (인기도/CF/카테고리)
│   │   ├── serving.py             # 추천 서빙 (end-to-end)
│   │   └── user_segmentation.py   # 유저 세그먼트 분류
│   │
│   ├── simulation/                # 시뮬레이션 모듈
│   │   ├── virtual_user.py        # 가상 유저 생성 & 평가
│   │   ├── virtual_user_enhanced.py # 강화된 가상 유저
│   │   ├── persona.py             # 유저 메타데이터 & 페르소나
│   │   ├── ab_test.py             # A/B 테스트 시뮬레이터
│   │   └── ollama_client.py       # Ollama LLM 클라이언트
│   │
│   ├── analysis/                  # 분석 모듈
│   │   └── statistical_tests.py   # 통계적 검정
│   │
│   └── utils/                     # 유틸리티
│       ├── db_init.py             # DB 초기화
│       ├── logging_utils.py       # 로깅 설정
│       └── validation.py          # 데이터 검증
│
├── scripts/                       # 실행 스크립트 (아래 상세 설명)
├── models/                        # 학습된 모델 저장소
│   └── artifacts/
│       └── purchase_ranker.pkl
├── logs/                          # 실행 로그 & A/B 테스트 결과
├── tests/                         # 테스트 코드
├── notebooks/                     # 분석 노트북
├── reports/                       # 리포트
└── .gitignore
```

---

## 설치 및 환경 설정

### 1. 사전 요구 사항

- Python 3.10+
- (선택) Ollama 서버 — LLM 기반 시뮬레이션 시 필요

### 2. 패키지 설치

```bash
pip install duckdb polars lightgbm mlflow streamlit pandas scipy scikit-learn tqdm optuna
```

### 3. 데이터 준비

`data/` 디렉토리에 H&M 데이터셋 파일을 배치합니다:

- `transactions_train.csv`
- `articles.csv`
- `customers.csv`

### 4. 환경 검증

```bash
python scripts/validate_environment.py
```

> Python 버전, 필수 패키지, DuckDB 연결, 데이터 파일 존재 여부를 자동으로 점검합니다.

---

## 파이프라인 실행 가이드

전체 파이프라인은 아래 순서로 실행합니다:

```bash
# Step 1. 환경 검증
python scripts/validate_environment.py

# Step 2. Feature 생성 (User + Item)
python scripts/process_data.py

# Step 3. 모델 학습
python scripts/train_model.py

# Step 4. (선택) 하이퍼파라미터 튜닝
python scripts/best_tuner.py

# Step 5. 배치 추론
python scripts/batch_inference.py --sample-size 100

# Step 6. A/B 테스트 시뮬레이션
python scripts/run_simulation.py --users 1000 --mode fast --ab-test

# Step 7. 결과 분석
python scripts/analyze_purchase_cycle.py
python scripts/calculate_sample_size.py
```

---

## Scripts 상세 설명

### 🔧 환경 & 검증

| 스크립트 | 설명 |
|---|---|
| [`validate_environment.py`](scripts/validate_environment.py) | 환경 설정 종합 검증 (Python 버전, 패키지, DuckDB, 데이터 파일, Feature 파일, 모델 디렉토리) |
| [`check_schema.py`](scripts/check_schema.py) | `user_features.parquet`의 스키마(컬럼 타입) 및 샘플 데이터 확인 |
| [`check_model_version.py`](scripts/check_model_version.py) | 모델 파일과 소스 코드의 수정 시각 비교 — 모델 재학습 필요 여부 판단 |
| [`check_ab_test_version.py`](scripts/check_ab_test_version.py) | A/B 테스트 결과와 모델 학습 시점 비교 — 테스트 결과가 최신 모델을 반영하는지 확인 |

### 📊 데이터 처리 & 분석

| 스크립트 | 설명 |
|---|---|
| [`process_data.py`](scripts/process_data.py) | **Feature Engineering 파이프라인** — `UserFeatureGenerator`와 `ItemFeatureGenerator`를 순차 실행하여 User/Item Feature를 Parquet으로 생성 |
| [`analyze_purchase_cycle.py`](scripts/analyze_purchase_cycle.py) | **구매 주기 분석** — 전체 유저의 평균 구매 주기 통계, 주기 분포(1주/2주/1개월/2개월/3개월+), 빈도별·최근 활동 유저별 분석 |
| [`extract_shopping_patterns.py`](scripts/extract_shopping_patterns.py) | **쇼핑 패턴 추출** — 나이-가격대 상관관계, 나이-카테고리 선호도, 구매 빈도 분포, 가격-빈도 상관관계, 카테고리 분포 분석 후 `data/shopping_patterns.json`으로 저장 |

### 🤖 모델 학습 & 추론

| 스크립트 | 설명 |
|---|---|
| [`train_model.py`](scripts/train_model.py) | **모델 학습 파이프라인** — `create_ranking_dataset` → User-based Train/Val Split → LightGBM LambdaRank 학습 → NDCG/MRR/HR 평가 → Feature Importance 출력 → `models/artifacts/purchase_ranker.pkl` 저장 |
| [`best_tuner.py`](scripts/best_tuner.py) | **하이퍼파라미터 자동 튜닝** — Optuna `LightGBMTuner`를 사용한 Hold-out 기반 튜닝 |
| [`batch_inference.py`](scripts/batch_inference.py) | **배치 추론** — 지정된 수의 유저에 대해 `RecommendationService`를 통한 Top-K 추천 생성 및 성능(처리 시간) 리포트 |

### 🧪 시뮬레이션 & A/B 테스트

| 스크립트 | 설명 |
|---|---|
| [`run_simulation.py`](scripts/run_simulation.py) | **메인 시뮬레이션 러너** — 두 가지 모드 지원: ① 기본 시뮬레이션 (가상 유저 페르소나 생성 → 추천 → 평가), ② A/B 테스트 (`--ab-test` 플래그로 Group A 인기 상품 vs Group B ML 추천 비교). `--mode fast`(룰 기반), `--mode llm`(LLM 기반) 선택 가능 |
| [`calculate_sample_size.py`](scripts/calculate_sample_size.py) | **A/B 테스트 표본 크기 계산** — 비율(CTR, 전환율) 및 평균값에 대한 통계적 검정력 분석. 유의수준, 검정력, MDE를 입력하여 필요한 최소 샘플 크기 산출 |
| [`verify_ab_simulation.py`](scripts/verify_ab_simulation.py) | A/B 시뮬레이션 검증 — 특정 유저에 대한 Group B(ML 추천) 시뮬레이션이 올바른 전략(`enhanced_fallback`)으로 라우팅되는지 확인 |
| [`verify_case_b.py`](scripts/verify_case_b.py) | Case B 후보군 검증 — 저빈도 유저(2~10회 구매)에 대한 카테고리 기반 후보군 생성 결과의 카테고리 분포 확인 |

### 🐛 디버깅 & 이슈 재현

| 스크립트 | 설명 |
|---|---|
| [`reproduce_issue.py`](scripts/reproduce_issue.py) | `load_user_metadata` 함수의 버그 수정 검증 — DuckDB에서 유저 메타데이터 로드 시 단건/배치 로드 모두 정상 작동하는지 테스트 |

---

## 핵심 모듈 (src)

### `src/data` — Feature Engineering

- **`UserFeatureGenerator`**: 유저별 구매 횟수, 평균 가격, 최근성(recency), 구매 빈도 등의 Feature 생성
- **`ItemFeatureGenerator`**: 아이템별 판매량, 카테고리, 가격대 등의 Feature 생성
- **`FeatureStore`**: Feature 파일 관리 및 로딩

### `src/models` — ML 모델

- **`create_ranking_dataset()`**: 유저별 긍정/부정 샘플을 조합한 Learning-to-Rank 데이터셋 생성
- **`PurchaseRanker`**: LightGBM LambdaRank 래퍼 (학습, 예측, 저장/로드)
- **`evaluate_ranking()`**: NDCG@K, MRR@K, HR@K 평가 메트릭
- **`CandidateGenerator`**: 인기도/협업필터링/카테고리 기반 다단계 후보군 생성
- **`RecommendationService`**: End-to-End 추천 서빙 (후보 생성 → 랭킹 → Top-K)
- **`UserSegmenter`**: 구매 패턴 기반 유저 세그먼트 분류

### `src/simulation` — 가상 유저 시뮬레이션

- **`VirtualUser`**: 가상 유저 페르소나 생성 및 추천 평가
- **`OllamaClient`**: 로컬 Ollama LLM 서버 연동
- **`ABTestSimulator`**: A/B 그룹 분할 및 시뮬레이션 실행
- **`persona`**: 유저 메타데이터 로드 및 페르소나 매핑

### `src/analysis` — 통계 분석

- **`statistical_tests`**: A/B 테스트 결과에 대한 통계적 유의성 검정

---

## 주요 CLI 옵션

### `run_simulation.py`

```bash
python scripts/run_simulation.py \
    --users 1000     # 시뮬레이션할 유저 수 (기본: 5)
    --mode fast      # fast(룰 기반) | llm(LLM 기반) | full(전체 LLM)
    --llm 1          # LLM 강제 사용(1) / 강제 미사용(0)
    --seed 42        # 유저 샘플링 seed
    --ab-test        # A/B 테스트 모드 활성화
```

### `batch_inference.py`

```bash
python scripts/batch_inference.py \
    --sample-size 100  # 처리할 유저 수 (기본: 100)
```

---

## 라이선스

이 프로젝트는 학습 및 연구 목적으로 제작되었습니다.
