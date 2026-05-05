# A/B 테스트 만족도 비교 로직 설명

## 📊 만족도 생성 과정

### 1. **VirtualUser에서 만족도 생성**

**위치**: `src/simulation/virtual_user.py`

#### Fallback 모드 (현재 사용 중)

```python
def _random_eval(self, n: int):
    """랜덤 평가 (Fallback 모드)"""
    purchase_count = random.randint(0, min(3, n))
    satisfaction = random.randint(2, 5)  # 2~5점 랜덤
    return purchase_count, satisfaction
```

**특징**:
- 만족도: **2~5점** 균등 분포
- 평균: 약 **3.5점**
- 추천 품질과 **무관** (완전 랜덤)

#### LLM 모드 (사용 불가)

```python
prompt = (
    f"You are a {self.persona['age']}-year-old {self.persona['gender']} shopper.\n"
    f"Style: {self.persona.get('style','casual')}. "
    f"Budget: {self.persona.get('budget','medium')}. "
    f"You received {n} product recommendations.\n"
    f"Return ONLY: Purchase: X, Satisfaction: Y (X is 0-{n}, Y is 1-5)."
)
```

**특징**:
- LLM이 페르소나 기반으로 만족도 생성
- 1~5점 범위
- 추천 품질 반영 가능

---

### 2. **AB Test에서 만족도 수집**

**위치**: `src/simulation/ab_test.py`

```python
def simulate_group_a(self, user_id: str, virtual_user: VirtualUser):
    # 인기 상품 추천
    popular_items = self.candidate_gen.generate_popularity_candidates(top_k=5)
    
    # 가상 유저 평가
    evaluation = virtual_user.evaluate_recommendations(popular_items)
    
    return {
        'clicked': clicked,
        'items': popular_items,
        'send_time': send_time,
        'num_items': len(popular_items),
        'purchase_count': evaluation.get('purchase_count', 0),
        'satisfaction': evaluation.get('satisfaction', 0)  # ← 만족도 수집
    }
```

**Group A (Control)**:
- 인기 상품 5개 추천
- 랜덤 만족도 (2~5점)

**Group B (Test)**:
- ML 모델 추천 5개
- 랜덤 만족도 (2~5점)

---

### 3. **분석 스크립트에서 만족도 비교**

**위치**: `scripts/analyze_ab_test.py`

```python
# Satisfaction Score
sat_a = df[df['group'] == 'A']['satisfaction'].mean()
sat_b = df[df['group'] == 'B']['satisfaction'].mean()
sat_diff = sat_b - sat_a

print(f"Group A Satisfaction: {sat_a:.3f} / 5.0")
print(f"Group B Satisfaction: {sat_b:.3f} / 5.0")
print(f"Difference: {sat_diff:+.3f}")
```

**비교 방법**:
- **평균값 비교** (t-test)
- Group A 평균 vs Group B 평균
- 차이의 통계적 유의성 검정

---

## 🔍 현재 문제점

### 1. **만족도가 추천 품질과 무관**

**현재**:
```python
satisfaction = random.randint(2, 5)  # 완전 랜덤!
```

**문제**:
- Group A와 Group B의 만족도가 **우연에 의해서만** 차이남
- ML 모델이 좋은 추천을 해도 만족도에 반영 안 됨
- **의미 없는 지표**

### 2. **통계적 검정 방법**

**현재 사용 (추정)**:
```python
# t-test 또는 Mann-Whitney U test
from scipy import stats
t_stat, p_value = stats.ttest_ind(
    df[df['group'] == 'A']['satisfaction'],
    df[df['group'] == 'B']['satisfaction']
)
```

**문제**:
- 랜덤 데이터를 검정해도 의미 없음
- 유의한 차이가 나올 확률 = 5% (우연)

---

## 💡 개선 방안

### 방안 1: 추천 품질 기반 만족도 (권장)

```python
def _calculate_satisfaction(self, recommendations: List[str], persona: Dict) -> int:
    """
    추천 품질 기반 만족도 계산
    
    고려 요소:
    1. 페르소나와 추천의 일치도
    2. 추천 다양성
    3. 추천 개수
    """
    base_satisfaction = 3  # 기본 만족도
    
    # 1. 추천 개수 (5개가 이상적)
    if len(recommendations) >= 5:
        base_satisfaction += 0.5
    elif len(recommendations) < 3:
        base_satisfaction -= 0.5
    
    # 2. 다양성 (모두 같은 카테고리면 감점)
    # (실제 구현 시 item features 활용)
    
    # 3. 랜덤 노이즈 추가 (현실성)
    noise = random.uniform(-0.5, 0.5)
    
    satisfaction = base_satisfaction + noise
    satisfaction = max(1, min(5, int(round(satisfaction))))
    
    return satisfaction
```

### 방안 2: 구매와 연동

```python
def _calculate_satisfaction(self, purchase_count: int, num_recommendations: int) -> int:
    """
    구매 행동 기반 만족도
    
    가정: 구매가 많을수록 만족도 높음
    """
    if purchase_count == 0:
        # 구매 안 함 → 낮은 만족도
        return random.randint(2, 3)
    elif purchase_count == 1:
        # 1개 구매 → 중간 만족도
        return random.randint(3, 4)
    else:
        # 2개 이상 구매 → 높은 만족도
        return random.randint(4, 5)
```

### 방안 3: 페르소나 기반 (가장 현실적)

```python
def _calculate_satisfaction(
    self, 
    recommendations: List[str], 
    persona: Dict,
    item_features: Dict
) -> int:
    """
    페르소나와 추천의 일치도 기반 만족도
    """
    match_score = 0
    
    for item_id in recommendations:
        item = item_features.get(item_id, {})
        
        # 카테고리 일치
        if item.get('category') in persona.get('categories', []):
            match_score += 1
        
        # 가격대 일치
        if item.get('price_tier') == persona.get('budget'):
            match_score += 0.5
        
        # 스타일 일치 (있다면)
        if item.get('style') == persona.get('style'):
            match_score += 0.5
    
    # 정규화 (0~5점)
    max_score = len(recommendations) * 2  # 완벽한 일치
    normalized_score = (match_score / max_score) * 3 + 2  # 2~5점 범위
    
    # 랜덤 노이즈
    noise = random.uniform(-0.3, 0.3)
    satisfaction = normalized_score + noise
    
    return max(1, min(5, int(round(satisfaction))))
```

---

## 📊 현재 A/B 테스트 결과 해석

**관찰된 결과**:
```
Group A Satisfaction: 3.502 / 5.0
Group B Satisfaction: 3.518 / 5.0
Difference: +0.016
```

**해석**:
- 차이: 0.016점 (거의 없음)
- 이는 **예상된 결과** (둘 다 랜덤이므로)
- 평균 3.5점 = random.randint(2, 5)의 기댓값
- 통계적으로 유의하지 않음 (p > 0.05)

**결론**:
- 현재 만족도 지표는 **의미 없음**
- 추천 품질과 무관한 랜덤 값

---

## 🎯 권장 조치

### 즉시 조치

1. **만족도 계산 로직 개선**
   - 구매 행동 기반 만족도 (방안 2)
   - 간단하고 직관적

2. **코드 수정**
   ```python
   # virtual_user.py의 _random_eval 수정
   def _random_eval(self, n: int):
       purchase_count = random.randint(0, min(3, n))
       
       # 구매 기반 만족도
       if purchase_count == 0:
           satisfaction = random.randint(2, 3)
       elif purchase_count == 1:
           satisfaction = random.randint(3, 4)
       else:
           satisfaction = random.randint(4, 5)
       
       return purchase_count, satisfaction
   ```

### 장기 조치

1. **페르소나 기반 만족도** (방안 3)
   - 추천과 페르소나의 일치도 계산
   - 더 현실적인 시뮬레이션

2. **LLM 모드 활성화**
   - Ollama 문제 해결 후
   - LLM이 추천 품질 평가

---

## 📝 결론

**현재 상태**:
- 만족도 = `random.randint(2, 5)` (완전 랜덤)
- 추천 품질과 **무관**
- Group A와 B의 차이 = **우연**

**개선 필요**:
- 구매 행동 기반 만족도 (최소)
- 페르소나 기반 만족도 (이상적)

**통계적 검정**:
- t-test 또는 Mann-Whitney U test
- 평균값 비교

---

**생성일**: 2026-01-15
**파일**: `src/simulation/virtual_user.py`, `scripts/analyze_ab_test.py`
