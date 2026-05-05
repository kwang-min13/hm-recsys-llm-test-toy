# A/B 테스트 페르소나 생성 문제 진단 리포트

## 📋 코드 분석 결과

### 1. 페르소나 생성 플로우

```
run_simulation.py (line 139)
  ↓
VirtualUser.generate_persona() (virtual_user.py line 44)
  ↓
├─ LLM 사용 가능? → ollama_client.generate() 호출
│   ↓
│   └─ JSON 파싱 성공? → 페르소나 반환
│       ↓ (실패시)
│       └─ _fallback_persona() 호출
│
└─ LLM 사용 불가 → _fallback_persona() 호출 (랜덤 생성)
```

### 2. 발견된 잠재적 문제점

#### ⚠️ 문제 1: 페르소나 재사용 이슈
**위치**: `run_simulation.py` line 126, 139

```python
# Line 126: VirtualUser 인스턴스 1개만 생성
vu = VirtualUser(ollama_client if use_llm else None)

# Line 139: 모든 유저가 동일한 vu 인스턴스 사용
for i, user_id in enumerate(user_ids, 1):
    persona = vu.generate_persona()  # ← 매번 새 페르소나 생성하지만...
```

**문제**:
- `VirtualUser` 인스턴스가 1개만 생성되어 모든 유저가 공유
- `self.persona` 속성이 덮어씌워짐
- 이전 유저의 페르소나가 다음 유저에게 영향을 줄 수 있음

**영향도**: 중간 (페르소나는 매번 새로 생성되지만, 상태 공유 가능)

#### ⚠️ 문제 2: LLM 연결 체크 캐싱
**위치**: `virtual_user.py` line 32-42

```python
def _is_llm_available(self) -> bool:
    if self._llm_available is None:
        try:
            self._llm_available = bool(self.ollama_client.check_connection())
        except Exception:
            self._llm_available = False
    return self._llm_available
```

**문제**:
- 연결 상태를 1회만 체크하고 캐시
- Ollama 서버가 중간에 다운되면 감지 못함
- 하지만 `ollama_client.generate()`에 재시도 로직 있음 (2회)

**영향도**: 낮음 (재시도 로직으로 커버됨)

#### ⚠️ 문제 3: Ollama API 엔드포인트 폴백
**위치**: `ollama_client.py` line 94-106

```python
# /api/generate가 404면 /api/chat로 폴백
if r.status_code == 404:
    logger.warning("POST /api/generate returned 404; falling back to /api/chat.")
    rc = self._post("/api/chat", json=payload_chat)
```

**문제**:
- Ollama 버전에 따라 `/api/generate` 미지원 가능
- 폴백 로직은 있지만, 로그만 남기고 사용자에게 알리지 않음

**영향도**: 낮음 (자동 폴백으로 해결)

#### ✅ 문제 4: 페르소나 생성 최적화 (이미 적용됨)
**위치**: `virtual_user.py` line 62-68

```python
response = self.ollama_client.generate(
    prompt,
    temperature=0.6,
    num_predict=140,  # ← 출력 길이 제한
    stop=["\\n\\n"]    # ← 불필요한 추가 문단 차단
)
```

**상태**: ✅ 이미 최적화되어 있음
- 출력 길이 제한으로 빠른 응답
- JSON만 출력하도록 프롬프트 설계

### 3. 권장 수정사항

#### 🔧 수정 1: VirtualUser 인스턴스 분리 (중요도: 높음)

**현재 코드** (`run_simulation.py` line 126-139):
```python
vu = VirtualUser(ollama_client if use_llm else None)

for i, user_id in enumerate(user_ids, 1):
    persona = vu.generate_persona()
    # ...
```

**수정 제안**:
```python
for i, user_id in enumerate(user_ids, 1):
    # 각 유저마다 새 VirtualUser 인스턴스 생성
    vu = VirtualUser(ollama_client if use_llm else None)
    persona = vu.generate_persona()
    # ...
```

**이유**:
- 페르소나 상태 완전 분리
- 메모리 사용량 증가는 미미 (VirtualUser는 경량)
- 더 안전한 멀티유저 시뮬레이션

#### 🔧 수정 2: 에러 처리 개선 (중요도: 중간)

**현재 코드** (`run_simulation.py` line 175-177):
```python
except Exception as e:
    logger.exception(f"시뮬레이션 실패 (user={user_id}, group={group}): {e}")
    continue
```

**수정 제안**:
```python
except Exception as e:
    logger.exception(f"시뮬레이션 실패 (user={user_id}, group={group}): {e}")
    # 실패한 유저도 기본값으로 기록 (데이터 손실 방지)
    results.append({
        'user_id': user_id,
        'group': group,
        'clicked': False,
        'send_time': 12,
        'num_items': 0,
        'purchase_count': 0,
        'satisfaction': 0,
        'persona_age': None,
        'persona_budget': None,
        'timestamp': pd.Timestamp.now(),
        'error': str(e)
    })
```

#### 🔧 수정 3: LLM 연결 상태 로깅 개선 (중요도: 낮음)

**현재 코드** (`run_simulation.py` line 102-105):
```python
if use_llm and conn_ok:
    logger.info("✓ Ollama 서버 연결 성공 (LLM 사용)")
else:
    logger.info("✓ LLM 미사용 모드 (룰 기반 평가)")
```

**수정 제안**:
```python
if use_llm and conn_ok:
    logger.info("✓ Ollama 서버 연결 성공 (LLM 사용)")
    logger.info(f"  Model: {ollama_client.model}")
    logger.info(f"  Base URL: {ollama_client.base_url}")
elif use_llm and not conn_ok:
    logger.warning("⚠ LLM 사용 요청되었으나 Ollama 연결 실패")
    logger.warning("  → 룰 기반 평가로 폴백")
else:
    logger.info("✓ LLM 미사용 모드 (룰 기반 평가)")
```

### 4. 테스트 권장사항

#### 테스트 1: 페르소나 생성 단독 테스트
```bash
python -c "from src.simulation.virtual_user import VirtualUser; vu = VirtualUser(None); print(vu.generate_persona())"
```

#### 테스트 2: Ollama 연결 테스트
```bash
python -c "from src.simulation.ollama_client import OllamaClient; client = OllamaClient(); print('Connected:', client.check_connection())"
```

#### 테스트 3: 소규모 A/B 테스트
```bash
python scripts/run_simulation.py --ab-test --users 10 --llm 0 --seed 42
```

### 5. 결론

**현재 코드 상태**: ✅ 대체로 양호
- 페르소나 생성 로직은 잘 설계됨
- LLM 폴백 메커니즘 존재
- 최적화 적용됨

**주요 개선 필요 사항**:
1. ⚠️ **VirtualUser 인스턴스 분리** (권장)
2. 에러 처리 개선 (선택)
3. 로깅 개선 (선택)

**예상 문제**:
- 페르소나 생성 자체는 문제 없음
- 다만 VirtualUser 재사용으로 인한 미묘한 상태 공유 가능성

---

**생성일**: 2026-01-15
**분석 대상**: `run_simulation.py`, `ab_test.py`, `virtual_user.py`, `ollama_client.py`
