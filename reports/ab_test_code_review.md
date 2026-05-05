# A/B 테스트 코드 체크 리포트

## 📋 검토 대상 파일

1. `src/simulation/ab_test.py` - A/B 테스트 시뮬레이터
2. `scripts/run_simulation.py` - A/B 테스트 실행 스크립트
3. `scripts/analyze_ab_test.py` - 결과 분석 스크립트
4. `src/simulation/virtual_user.py` - 가상 유저 로직

---

## ✅ 정상 작동하는 부분

### 1. **기본 구조**
- Group A (Control): 인기 상품 추천 ✅
- Group B (Test): ML 모델 추천 ✅
- 50:50 랜덤 할당 ✅

### 2. **데이터 수집**
- CTR (Click-Through Rate) ✅
- 구매 수 (Purchase Count) ✅
- 만족도 (Satisfaction) ✅
- 발송 시간 (Send Time) ✅

### 3. **통계 분석**
- Chi-Square Test (CTR) ✅
- T-Test (구매 수, 만족도) ✅
- 시각화 (그래프) ✅

---

## ⚠️ 발견된 문제점

### 🔴 **문제 1: VirtualUser 인스턴스 재사용**

**위치**: `scripts/run_simulation.py` line 126

```python
# 현재 코드 (문제)
vu = VirtualUser(ollama_client if use_llm else None)  # ← 루프 밖에서 생성

for i in range(num_users):
    user_id = generate_user_id(i, seed)
    persona = vu.generate_persona()  # ← 같은 인스턴스 재사용
```

**문제점**:
- 모든 유저가 **같은 VirtualUser 인스턴스** 사용
- `self.persona`가 덮어씌워짐
- 이전 유저의 페르소나가 다음 유저에게 영향

**영향**:
- 페르소나 생성은 정상 (매번 새로 생성)
- 하지만 `evaluate_recommendations`에서 `self.persona` 참조 시 문제 가능

**해결책**:
```python
# 수정된 코드
for i in range(num_users):
    user_id = generate_user_id(i, seed)
    vu = VirtualUser(ollama_client if use_llm else None)  # ← 루프 안에서 생성
    persona = vu.generate_persona()
```

**상태**: ⚠️ **이미 수정됨** (이전 세션에서 수정)

---

### 🟡 **문제 2: 만족도가 추천 품질과 무관**

**위치**: `src/simulation/virtual_user.py` line 294

```python
def _random_eval(self, n: int):
    purchase_count = random.randint(0, min(3, n))
    satisfaction = random.randint(2, 5)  # ← 완전 랜덤!
    return purchase_count, satisfaction
```

**문제점**:
- 만족도가 추천 품질과 **무관**
- Group A와 B의 차이가 **우연**에 의존
- 의미 없는 지표

**영향**:
- 만족도 비교 결과가 신뢰할 수 없음
- ML 모델 성능을 제대로 평가 못함

**해결책**:
```python
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

**상태**: ⚠️ **수정 필요**

---

### 🟡 **문제 3: 클릭 로직이 단순함**

**위치**: `src/simulation/ab_test.py` line 68, 98

```python
# 클릭 여부: 구매 예상이 1개 이상이면 클릭
clicked = evaluation.get('purchase_count', 0) > 0
```

**문제점**:
- 클릭 = 구매 예상 여부
- 너무 단순한 로직
- 실제로는 클릭했지만 구매 안 할 수도 있음

**현실적 시나리오**:
- 클릭 확률: 70-80%
- 클릭 후 구매 확률: 50%
- 전체 구매 확률: 35-40%

**해결책**:
```python
# 1단계: 클릭 여부 (확률적)
click_probability = 0.75  # 75% 클릭
clicked = random.random() < click_probability

# 2단계: 클릭했을 때만 구매 평가
if clicked:
    evaluation = virtual_user.evaluate_recommendations(rec_items)
else:
    evaluation = {'purchase_count': 0, 'satisfaction': 0}
```

**상태**: ⚠️ **개선 권장** (현재 로직도 작동은 함)

---

### 🟡 **문제 4: Group A의 발송 시간이 랜덤**

**위치**: `src/simulation/ab_test.py` line 58

```python
def simulate_group_a(self, user_id: str, virtual_user: VirtualUser):
    # 2. 랜덤 발송 시간 (9시~21시)
    send_time = random.randint(9, 21)
```

**문제점**:
- Group A는 랜덤 시간 (9-21시)
- Group B는 ML 모델이 예측한 최적 시간
- **공정한 비교가 아님**

**현실적 대안**:
1. **고정 시간 사용**: Group A도 특정 시간 (예: 12시)
2. **평균 시간 사용**: Group A도 전체 평균 시간
3. **현재 유지**: 랜덤 vs 최적 비교

**상태**: ⚠️ **설계 의도 확인 필요**

---

### 🟢 **문제 5: DuckDB 파일 잠금 (해결됨)**

**위치**: `src/models/candidate_generation.py`, `src/data/feature_store.py`

**문제점**:
- 파일 기반 DuckDB 사용 시 동시 접근 불가
- 여러 프로세스가 충돌

**해결책**: ✅ **이미 수정됨**
- 메모리 데이터베이스 사용
- 읽기 전용 모드로 파일 attach

---

### 🟡 **문제 6: 샘플 크기 부족**

**현재 설정**: 1,000명 (기본값)

**문제점**:
- 4.05%p 차이를 감지하려면 **1,525명/그룹** 필요
- 현재 511명/489명 → **부족**

**해결책**:
```bash
# 최소 3,000명 권장
python scripts/run_simulation.py --ab-test --users 3000 --llm 0 --seed 42

# 표준 검증 (10,000명)
python scripts/run_simulation.py --ab-test --users 10000 --llm 0 --seed 42
```

**상태**: ⚠️ **실행 시 조정 필요**

---

## 🔍 코드 품질 체크

### 1. **에러 처리**

✅ **양호**:
```python
try:
    result = ab_simulator.simulate_group_a(user_id, vu)
except Exception as e:
    logger.error(f"시뮬레이션 실패 (user={user_id}, group=A): {e}")
    continue
```

### 2. **로깅**

✅ **양호**:
```python
logger.info(f"--- 시뮬레이션 {i+1}/{num_users} (Group {group}) ---")
logger.info(f"페르소나: {persona['age']}세 {persona['gender']}, ...")
```

### 3. **시드 설정**

✅ **양호**:
```python
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
```

### 4. **결과 저장**

✅ **양호**:
```python
results_df.to_csv(output_path, index=False)
logger.info(f"결과 저장: {output_path}")
```

---

## 📊 데이터 흐름 검증

```
1. run_simulation.py
   ↓
2. VirtualUser.generate_persona()
   → 페르소나 생성 (나이, 성별, 스타일, 예산, 빈도, 카테고리)
   ↓
3. ABTestSimulator.simulate_group_a/b()
   → Group A: 인기 상품 5개 + 랜덤 시간
   → Group B: ML 추천 5개 + 최적 시간
   ↓
4. VirtualUser.evaluate_recommendations()
   → 구매 수 (0-3개)
   → 만족도 (2-5점) ← 랜덤!
   ↓
5. 클릭 여부 결정
   → clicked = purchase_count > 0
   ↓
6. 결과 저장
   → logs/ab_test_results.csv
   ↓
7. analyze_ab_test.py
   → CTR, 구매율, 만족도 비교
   → 통계적 유의성 검정
   → 시각화
```

**검증 결과**: ✅ **데이터 흐름 정상**

---

## 🎯 우선순위별 개선 사항

### 🔴 **높음** (즉시 수정 권장)

1. **만족도 로직 개선**
   - 구매 기반 만족도로 변경
   - 추천 품질 반영

2. **샘플 크기 증가**
   - 최소 3,000명 사용
   - 통계적 유의성 확보

### 🟡 **중간** (개선 권장)

3. **클릭 로직 개선**
   - 확률적 클릭 모델
   - 클릭 ≠ 구매 분리

4. **Group A 발송 시간**
   - 설계 의도 확인
   - 공정한 비교 고려

### 🟢 **낮음** (선택 사항)

5. **LLM 모드 활성화**
   - Ollama 문제 해결 후
   - 더 현실적인 페르소나

6. **추가 지표 수집**
   - 추천 다양성
   - 카테고리 분포
   - 가격대 분포

---

## 📝 코드 개선 제안

### 제안 1: 만족도 로직 개선

```python
# src/simulation/virtual_user.py
def _random_eval(self, n: int):
    purchase_count = random.randint(0, min(3, n))
    
    # 구매 기반 만족도
    if purchase_count == 0:
        satisfaction = random.randint(2, 3)  # 낮음
    elif purchase_count == 1:
        satisfaction = random.randint(3, 4)  # 중간
    else:
        satisfaction = random.randint(4, 5)  # 높음
    
    return purchase_count, satisfaction
```

### 제안 2: 클릭 로직 개선

```python
# src/simulation/ab_test.py
def simulate_group_a(self, user_id: str, virtual_user: VirtualUser):
    popular_items = self.candidate_gen.generate_popularity_candidates(top_k=5)
    send_time = random.randint(9, 21)
    
    # 1단계: 클릭 여부 (확률적)
    click_probability = 0.75
    clicked = random.random() < click_probability
    
    # 2단계: 클릭했을 때만 평가
    if clicked:
        evaluation = virtual_user.evaluate_recommendations(popular_items)
    else:
        evaluation = {'purchase_count': 0, 'satisfaction': 0}
    
    return {
        'clicked': clicked,
        'items': popular_items,
        'send_time': send_time,
        'num_items': len(popular_items),
        'purchase_count': evaluation.get('purchase_count', 0),
        'satisfaction': evaluation.get('satisfaction', 0)
    }
```

---

## ✅ 결론

**전반적 평가**: 🟢 **양호**

**강점**:
- 기본 구조 탄탄
- 에러 처리 잘 됨
- 로깅 충분
- 통계 분석 적절

**개선 필요**:
- 만족도 로직 (우선순위 높음)
- 샘플 크기 (우선순위 높음)
- 클릭 로직 (우선순위 중간)

**즉시 실행 가능**:
```bash
# 개선된 설정으로 A/B 테스트 실행
python scripts/run_simulation.py --ab-test --users 10000 --llm 0 --seed 42
```

---

**검토일**: 2026-01-15
**검토자**: AI Assistant
**검토 파일**: 4개 (ab_test.py, run_simulation.py, analyze_ab_test.py, virtual_user.py)
