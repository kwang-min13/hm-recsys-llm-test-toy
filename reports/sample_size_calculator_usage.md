# A/B 테스트 표본 크기 계산기 사용법

## 🎯 목적

통계적으로 유의미한 A/B 테스트 결과를 얻기 위해 필요한 최소 샘플 크기를 계산합니다.

---

## 🚀 빠른 시작

### 기본 사용법

```bash
python scripts/calculate_sample_size.py
```

이 명령어는 4가지 예시 시나리오를 실행합니다:
1. 기본 시나리오 (5% 개선 감지)
2. 작은 효과 감지 (2% 개선)
3. 높은 검정력 (90%)
4. 현재 A/B 테스트 결과 분석

---

## 📊 주요 함수

### 1. `calculate_sample_size_proportion()`

**용도**: 비율(CTR, 전환율 등) 비교

**예시**:
```python
from scripts.calculate_sample_size import calculate_sample_size_proportion

result = calculate_sample_size_proportion(
    baseline_rate=0.75,              # Control 그룹 CTR: 75%
    minimum_detectable_effect=0.05,  # 감지하려는 차이: 5%p
    alpha=0.05,                      # 유의수준: 5%
    power=0.80                       # 검정력: 80%
)

print(f"필요한 샘플: {result['sample_size_per_group']:,}명/그룹")
```

### 2. `calculate_sample_size_mean()`

**용도**: 평균값(만족도, 구매액 등) 비교

**예시**:
```python
from scripts.calculate_sample_size import calculate_sample_size_mean

result = calculate_sample_size_mean(
    baseline_mean=3.5,               # Control 그룹 평균 만족도
    baseline_std=1.0,                # 표준편차
    minimum_detectable_effect=0.2,   # 감지하려는 차이
    alpha=0.05,
    power=0.80
)

print(f"필요한 샘플: {result['sample_size_per_group']:,}명/그룹")
```

### 3. `calculate_sample_size_for_ab_test()`

**용도**: 종합적인 A/B 테스트 분석 (권장)

**예시**:
```python
from scripts.calculate_sample_size import calculate_sample_size_for_ab_test, print_sample_size_report

result = calculate_sample_size_for_ab_test(
    baseline_ctr=0.75,    # 현재 CTR
    target_lift=0.05,     # 목표 개선율: 5%
    alpha=0.05,
    power=0.80
)

print_sample_size_report(result)
```

---

## 🔧 파라미터 설명

### 필수 파라미터

| 파라미터 | 설명 | 예시 |
|---------|------|------|
| `baseline_rate` | Control 그룹의 현재 비율 | 0.75 (75%) |
| `baseline_mean` | Control 그룹의 현재 평균 | 3.5 |
| `minimum_detectable_effect` | 감지하려는 최소 차이 | 0.05 (5%p 또는 5% 개선) |

### 선택 파라미터

| 파라미터 | 설명 | 기본값 | 권장값 |
|---------|------|--------|--------|
| `alpha` | 유의수준 (Type I Error) | 0.05 | 0.05 (5%) |
| `power` | 검정력 (1 - Type II Error) | 0.80 | 0.80-0.90 |
| `two_tailed` | 양측 검정 여부 | True | True |

---

## 📈 실전 예시

### 예시 1: 현재 성능 기반 계산

```python
# 현재 시스템 성능
current_ctr = 0.75  # 75% CTR

# 10% 개선을 목표로 함
target_improvement = 0.10

result = calculate_sample_size_for_ab_test(
    baseline_ctr=current_ctr,
    target_lift=target_improvement,
    alpha=0.05,
    power=0.80
)

print_sample_size_report(result)
```

**출력**:
```
RECOMMENDED SAMPLE SIZE
  Per Group: 4,951 users
  Total: 9,902 users
```

### 예시 2: 작은 효과 감지

```python
# 2% 개선도 감지하고 싶음
result = calculate_sample_size_for_ab_test(
    baseline_ctr=0.75,
    target_lift=0.02,  # 2% 개선
    alpha=0.05,
    power=0.80
)

print_sample_size_report(result)
```

**출력**:
```
RECOMMENDED SAMPLE SIZE
  Per Group: 65,664 users
  Total: 131,328 users
```

### 예시 3: 높은 신뢰도

```python
# 90% 검정력으로 5% 개선 감지
result = calculate_sample_size_for_ab_test(
    baseline_ctr=0.75,
    target_lift=0.05,
    alpha=0.05,
    power=0.90  # 90% 검정력
)

print_sample_size_report(result)
```

**출력**:
```
RECOMMENDED SAMPLE SIZE
  Per Group: 14,143 users
  Total: 28,286 users
```

---

## 💡 해석 가이드

### 결과 읽기

```python
result = {
    'recommended_sample_size_per_group': 10565,
    'recommended_total_sample_size': 21130,
    'ctr_analysis': {...},
    'conversion_analysis': {...},
    'satisfaction_analysis': {...}
}
```

**의미**:
- **10,565명/그룹** 필요 (총 21,130명)
- CTR, 구매 전환율, 만족도 중 **가장 큰 값** 선택
- 가장 보수적인 추정 (안전한 선택)

### 지표별 분석

1. **CTR (Click-Through Rate)**
   - 가장 빠르게 측정 가능
   - 샘플 크기 상대적으로 작음

2. **구매 전환율**
   - 가장 중요한 비즈니스 지표
   - **샘플 크기 가장 큼** (보통 이 값 사용)

3. **만족도**
   - 장기적 성공 지표
   - 샘플 크기 중간

---

## ⚠️ 주의사항

### 1. 실제 vs 예상

계산된 샘플 크기는 **최소값**입니다:
- 실제로는 10-20% 더 많이 수집 권장
- 이탈, 오류 등을 고려

### 2. 효과 크기 선택

너무 작은 효과를 감지하려면:
- 매우 큰 샘플 필요
- 비용 대비 효과 고려

**권장**:
- 비즈니스적으로 의미 있는 최소 효과 선택
- 보통 5-10% 개선이 적절

### 3. 검정력 vs 샘플 크기

| 검정력 | 샘플 크기 | 권장 상황 |
|--------|-----------|-----------|
| 70% | 작음 | 예비 테스트 |
| 80% | 중간 | **표준** ⭐ |
| 90% | 큼 | 중요한 결정 |
| 95% | 매우 큼 | 매우 중요한 결정 |

---

## 🎯 실무 적용

### 시나리오 1: 빠른 검증

**목표**: ML 모델이 확실히 나은지 빠르게 확인

```python
result = calculate_sample_size_for_ab_test(
    baseline_ctr=0.75,
    target_lift=0.10,  # 10% 개선 (큰 효과)
    power=0.70         # 낮은 검정력
)
# 결과: ~2,000명
```

### 시나리오 2: 표준 검증

**목표**: 신뢰할 수 있는 결과

```python
result = calculate_sample_size_for_ab_test(
    baseline_ctr=0.75,
    target_lift=0.05,  # 5% 개선
    power=0.80         # 표준 검정력
)
# 결과: ~10,000명
```

### 시나리오 3: 엄격한 검증

**목표**: 프로덕션 배포 결정

```python
result = calculate_sample_size_for_ab_test(
    baseline_ctr=0.75,
    target_lift=0.05,  # 5% 개선
    power=0.90         # 높은 검정력
)
# 결과: ~14,000명
```

---

## 📚 추가 리소스

### 관련 파일

- `scripts/calculate_sample_size.py`: 계산기 스크립트
- `reports/sample_size_guide.md`: 상세 가이드
- `scripts/run_simulation.py`: A/B 테스트 실행

### 참고 문헌

- [Statistical Power Analysis](https://en.wikipedia.org/wiki/Power_of_a_test)
- [A/B Testing Calculator](https://www.evanmiller.org/ab-testing/sample-size.html)
- [Effect Size](https://en.wikipedia.org/wiki/Effect_size)

---

**생성일**: 2026-01-15
**버전**: 1.0
