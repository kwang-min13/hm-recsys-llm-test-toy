# Fallback 모드 페르소나 생성 로직 설명

## 📋 전체 플로우

```python
VirtualUser.generate_persona()
  ↓
1. 기본 속성 랜덤 생성 (항상 실행)
   - age: random.randint(18, 65)
   - gender: random.choice(["Male", "Female", "Non-binary"])
  ↓
2. LLM 사용 가능 여부 체크
   ↓
   [LLM 사용 불가 또는 --llm 0]
   ↓
3. _fallback_persona() 호출
   ↓
4. 최종 페르소나 반환
   - {age, gender, style, frequency, budget, categories}
```

---

## 🎯 Fallback 모드 상세 로직

### 1. 기본 속성 생성 (항상 실행)

**위치**: `virtual_user.py` line 46-47

```python
age = random.randint(18, 65)        # 18세 ~ 65세 랜덤
gender = random.choice([            # 성별 랜덤 선택
    "Male", 
    "Female", 
    "Non-binary"
])
```

**특징**:
- LLM 사용 여부와 무관하게 항상 실행
- 완전 랜덤 생성
- 균등 분포

---

### 2. LLM 사용 가능 여부 체크

**위치**: `virtual_user.py` line 49

```python
if self._is_llm_available():
    # LLM 모드 (현재 사용 불가)
    ...
else:
    # Fallback 모드 (현재 사용 중)
    persona_details = self._fallback_persona()
```

**`_is_llm_available()` 조건**:
- `ollama_client`가 `None`이면 → `False`
- `--llm 0` 옵션 사용 시 → `ollama_client = None` → `False`
- Ollama 연결 실패 시 → `False`

---

### 3. `_fallback_persona()` 함수

**위치**: `virtual_user.py` line 115-127

```python
def _fallback_persona(self) -> Dict[str, Any]:
    """LLM 실패 시 대체 페르소나"""
    
    # 1. 선택 가능한 옵션 정의
    styles = ["casual", "formal", "sporty", "trendy", "vintage"]
    frequencies = ["weekly", "monthly", "occasionally"]
    budgets = ["low", "medium", "high"]
    categories = ["tops", "bottoms", "dresses", "shoes", "accessories", "outerwear"]
    
    # 2. 랜덤 선택
    return {
        "style": random.choice(styles),           # 5개 중 1개
        "frequency": random.choice(frequencies),  # 3개 중 1개
        "budget": random.choice(budgets),         # 3개 중 1개
        "categories": random.sample(categories, 2) # 6개 중 2개 (중복 없음)
    }
```

---

## 📊 생성 가능한 페르소나 조합

### 속성별 옵션 수

| 속성 | 옵션 수 | 가능한 값 |
|------|---------|-----------|
| **age** | 48개 | 18 ~ 65 |
| **gender** | 3개 | Male, Female, Non-binary |
| **style** | 5개 | casual, formal, sporty, trendy, vintage |
| **frequency** | 3개 | weekly, monthly, occasionally |
| **budget** | 3개 | low, medium, high |
| **categories** | 15개 | 6개 중 2개 조합 (6C2 = 15) |

### 총 조합 수

```
총 조합 = 48 × 3 × 5 × 3 × 3 × 15
       = 97,200 가지
```

**결론**: 충분히 다양한 페르소나 생성 가능!

---

## 🎲 랜덤 분포 특성

### 1. 균등 분포 (Uniform Distribution)

모든 속성이 **균등 분포**를 따름:
- 각 옵션이 선택될 확률이 동일
- 예: `budget`의 경우
  - low: 33.3%
  - medium: 33.3%
  - high: 33.3%

### 2. 독립성 (Independence)

각 속성은 **독립적**으로 선택됨:
- `age`와 `budget`는 상관관계 없음
- `gender`와 `style`은 상관관계 없음
- 실제 현실과는 다를 수 있음 (예: 젊은 사람이 trendy 선호)

---

## 💡 Fallback 모드 vs LLM 모드

| 특징 | Fallback 모드 | LLM 모드 |
|------|---------------|----------|
| **속도** | 즉시 (0.000s) | 느림 (14.4s) |
| **다양성** | 97,200 조합 | 무한대 |
| **현실성** | 낮음 (독립 분포) | 높음 (상관관계 반영) |
| **일관성** | 높음 (결정적) | 낮음 (LLM 변동성) |
| **비용** | 무료 | Ollama 서버 필요 |

---

## 🔍 예시 페르소나

### 예시 1
```python
{
    "age": 28,
    "gender": "Female",
    "style": "trendy",
    "frequency": "weekly",
    "budget": "high",
    "categories": ["dresses", "shoes"]
}
```

### 예시 2
```python
{
    "age": 52,
    "gender": "Male",
    "style": "casual",
    "frequency": "occasionally",
    "budget": "low",
    "categories": ["tops", "outerwear"]
}
```

### 예시 3
```python
{
    "age": 35,
    "gender": "Non-binary",
    "style": "vintage",
    "frequency": "monthly",
    "budget": "medium",
    "categories": ["accessories", "bottoms"]
}
```

---

## ⚙️ 사용 방법

### A/B 테스트에서 Fallback 모드 활성화

```bash
# --llm 0 옵션으로 Fallback 모드 강제 사용
python scripts/run_simulation.py --ab-test --users 1000 --llm 0 --seed 42
```

**효과**:
- `VirtualUser(ollama_client=None)` 생성
- `_is_llm_available()` → `False`
- `_fallback_persona()` 호출
- 즉시 페르소나 생성

---

## 🎯 Fallback 모드의 장단점

### ✅ 장점

1. **초고속**: 0.000s (즉시 완료)
2. **안정적**: Ollama 서버 불필요
3. **재현 가능**: `seed` 설정으로 동일한 결과
4. **충분한 다양성**: 97,200가지 조합
5. **A/B 테스트에 적합**: 빠른 대규모 시뮬레이션

### ⚠️ 단점

1. **현실성 부족**: 속성 간 상관관계 없음
   - 예: 65세 + trendy + high budget (비현실적 조합 가능)
2. **단순한 분포**: 균등 분포만 사용
3. **맥락 없음**: 실제 쇼핑 패턴 미반영

---

## 🚀 권장 사항

### A/B 테스트 목적

**Fallback 모드 사용 권장** ✅
- 빠른 실행 (1,000명 = 수 초)
- 충분한 다양성
- 재현 가능성

### 프로덕션 시뮬레이션

**LLM 모드 고려** (Ollama 문제 해결 후)
- 더 현실적인 페르소나
- 상관관계 반영
- 단, 속도 느림 (소규모 테스트만)

---

**생성일**: 2026-01-15
**파일**: `src/simulation/virtual_user.py`
**함수**: `_fallback_persona()` (line 115-127)
