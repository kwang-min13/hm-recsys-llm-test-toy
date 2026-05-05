# 페르소나 생성 성능 분석 리포트

## 🔴 **심각한 문제 발견!**

### 테스트 결과 요약

| 항목 | 결과 | 상태 |
|------|------|------|
| **Ollama 연결** | 성공 (2.03s) | ✅ |
| **API 엔드포인트** | 404 에러 | 🔴 |
| **LLM 응답 속도** | 13.8s | 🔴 매우 느림 |
| **페르소나 생성 (LLM)** | 평균 14.4s | 🔴 매우 느림 |
| **페르소나 생성 (Fallback)** | 0.000s | ✅ 완벽 |

---

## 🔍 **문제 1: API 엔드포인트 404 에러**

### 현상
```
POST /api/generate → 404 Not Found
POST /api/chat     → 404 Not Found
```

### 원인 분석

1. **Ollama 버전 문제**
   - 연결은 성공 (`/api/version` 또는 `/api/tags` 응답)
   - 하지만 `/api/generate`와 `/api/chat` 모두 404
   - 이는 **Ollama 버전이 매우 오래되었거나** 설정 문제일 가능성

2. **엔드포인트 경로 문제**
   - 일부 Ollama 버전은 다른 경로 사용 가능
   - 또는 모델이 로드되지 않음

### 해결 방법

#### 방법 1: Ollama 버전 확인 및 업데이트
```bash
# 현재 버전 확인
ollama --version

# 최신 버전으로 업데이트
# Windows: 공식 사이트에서 최신 설치 파일 다운로드
# https://ollama.ai/download
```

#### 방법 2: 모델 재설치
```bash
# llama3 모델 재설치
ollama pull llama3

# 모델 목록 확인
ollama list
```

#### 방법 3: Ollama 서버 재시작
```bash
# 서비스 중지 후 재시작
# Windows: 작업 관리자에서 Ollama 프로세스 종료 후 재시작
```

#### 방법 4: API 엔드포인트 수정 (코드 변경)
현재 코드는 이미 폴백 로직이 있지만, 더 나은 에러 처리가 필요합니다.

---

## 🔍 **문제 2: 페르소나 생성 속도 매우 느림**

### 측정 결과

```
LLM 모드:
- Attempt 1: 15.761s
- Attempt 2: 13.723s
- Attempt 3: 13.754s
- Average: 14.413s  ← 🔴 매우 느림!

Fallback 모드:
- Average: 0.000s   ← ✅ 즉시 완료
```

### 원인

1. **API 404 에러로 인한 재시도**
   - 각 요청마다 3번 재시도 (retries=2)
   - 재시도 간격: 0.5s, 1.0s
   - 총 대기 시간: ~1.5s per attempt
   - 3번 재시도 = 약 4.5s 추가 지연

2. **Ollama 서버 응답 느림**
   - API가 작동해도 13.8s 소요
   - 이는 비정상적으로 느림 (정상: 1-3s)

3. **누적 효과**
   - 1,000명 유저 시뮬레이션 시: 14.4s × 1,000 = **4시간!**

### 해결책

#### ✅ **즉시 적용 가능: Fallback 모드 사용**

```bash
# A/B 테스트 실행 시 LLM 비활성화
python scripts/run_simulation.py --ab-test --users 1000 --llm 0 --seed 42
```

**장점**:
- 즉시 실행 (0.000s per user)
- 1,000명 시뮬레이션: 수 초 내 완료
- 페르소나는 여전히 다양하게 생성됨 (랜덤)

**단점**:
- LLM 기반 페르소나보다 덜 현실적
- 하지만 A/B 테스트 목적으로는 충분

---

## 🔍 **문제 3: API 연결 체크 로직 개선 필요**

### 현재 문제

`check_connection()`은 `/api/version` 또는 `/api/tags`만 체크:
```python
def check_connection(self) -> bool:
    for path in ("/api/version", "/api/tags"):
        try:
            r = self._get(path, timeout=3)
            if r.status_code == 200:
                return True  # ← 연결 성공으로 판단
        except Exception:
            continue
    return False
```

하지만 실제 사용하는 `/api/generate`와 `/api/chat`는 404!

### 개선 방안

```python
def check_connection(self) -> bool:
    """실제 사용할 엔드포인트 체크"""
    # 1. 기본 연결 체크
    for path in ("/api/version", "/api/tags"):
        try:
            r = self._get(path, timeout=3)
            if r.status_code == 200:
                break
        except Exception:
            continue
    else:
        return False  # 기본 연결 실패
    
    # 2. 실제 API 엔드포인트 체크
    try:
        r = self._post("/api/generate", json={
            "model": self.model,
            "prompt": "test",
            "stream": False,
            "options": {"num_predict": 1}
        }, timeout=5)
        
        if r.status_code == 200:
            return True
        elif r.status_code == 404:
            # /api/chat 시도
            r = self._post("/api/chat", json={
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
                "options": {"num_predict": 1}
            }, timeout=5)
            return r.status_code == 200
    except Exception:
        pass
    
    return False
```

---

## 📊 **성능 비교**

| 시나리오 | 1명당 시간 | 1,000명 총 시간 | 상태 |
|---------|-----------|----------------|------|
| **LLM 모드 (현재)** | 14.4s | **4시간** | 🔴 사용 불가 |
| **Fallback 모드** | 0.000s | **수 초** | ✅ 권장 |
| **LLM 모드 (정상)** | 2-3s | 33-50분 | ⚠️ 느림 |

---

## 🎯 **권장 조치**

### 1. **즉시 조치** (A/B 테스트 실행)

```bash
# Fallback 모드로 A/B 테스트 실행
python scripts/run_simulation.py --ab-test --users 1000 --llm 0 --seed 42
```

### 2. **Ollama 문제 해결** (장기)

1. Ollama 버전 확인 및 업데이트
2. llama3 모델 재설치
3. Ollama 서버 재시작
4. API 엔드포인트 테스트

### 3. **코드 개선** (선택)

1. `check_connection()` 로직 개선
2. 타임아웃 설정 최적화
3. 재시도 로직 조정

---

## 📝 **테스트 명령어**

### Ollama 상태 확인
```bash
# 버전 확인
ollama --version

# 모델 목록
ollama list

# 모델 테스트
ollama run llama3 "Hello"
```

### 성능 테스트 재실행
```bash
python scripts/test_persona_performance.py
```

### A/B 테스트 실행 (Fallback 모드)
```bash
python scripts/run_simulation.py --ab-test --users 1000 --llm 0 --seed 42
```

---

## ✅ **결론**

**현재 상황**:
- 🔴 Ollama API가 404 에러 반환
- 🔴 LLM 모드는 14.4s/user로 매우 느림
- ✅ Fallback 모드는 즉시 작동

**즉시 해결책**:
- **Fallback 모드 사용** (`--llm 0`)
- 1,000명 A/B 테스트를 수 초 내 완료 가능

**장기 해결책**:
- Ollama 버전 업데이트 및 재설정
- API 엔드포인트 문제 해결

---

**생성일**: 2026-01-15 12:57
**테스트 환경**: Windows, Ollama (localhost:11434)
