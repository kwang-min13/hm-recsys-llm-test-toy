"""
Persona Generation Performance Test

페르소나 생성 속도 및 API 연결 테스트
"""

import time
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.virtual_user import VirtualUser
from src.simulation.ollama_client import OllamaClient

print("=" * 80)
print("PERSONA GENERATION PERFORMANCE TEST")
print("=" * 80)

# Test 1: Ollama 연결 테스트
print("\n[Test 1] Ollama Connection Test")
print("-" * 80)

ollama_client = OllamaClient()

print(f"Base URL: {ollama_client.base_url}")
print(f"Model: {ollama_client.model}")
print(f"Timeout: {ollama_client.timeout_sec}s")

print("\nTesting connection...")
t0 = time.time()
try:
    connected = ollama_client.check_connection()
    t1 = time.time()
    print(f"Connection check time: {t1-t0:.3f}s")
    print(f"Result: {'CONNECTED' if connected else 'NOT CONNECTED'}")
except Exception as e:
    t1 = time.time()
    print(f"Connection check time: {t1-t0:.3f}s")
    print(f"Error: {e}")
    connected = False

# Test 2: LLM 응답 속도 테스트
if connected:
    print("\n[Test 2] LLM Response Speed Test")
    print("-" * 80)
    
    test_prompt = "Say 'Hello' in one word."
    
    print(f"Prompt: {test_prompt}")
    print("Sending request...")
    
    t0 = time.time()
    try:
        response = ollama_client.generate(
            test_prompt,
            temperature=0.3,
            num_predict=10,
            stop=["\n"]
        )
        t1 = time.time()
        
        print(f"Response time: {t1-t0:.3f}s")
        print(f"Response: {response}")
        
        if t1 - t0 > 5:
            print("[WARNING] Response time > 5s - This is SLOW!")
        elif t1 - t0 > 2:
            print("[WARNING] Response time > 2s - Consider optimization")
        else:
            print("[OK] Response time is acceptable")
            
    except Exception as e:
        t1 = time.time()
        print(f"Response time: {t1-t0:.3f}s")
        print(f"Error: {e}")
else:
    print("\n[Test 2] SKIPPED - Ollama not connected")

# Test 3: 페르소나 생성 속도 테스트 (LLM 사용)
print("\n[Test 3] Persona Generation Speed Test (WITH LLM)")
print("-" * 80)

if connected:
    vu_llm = VirtualUser(ollama_client)
    
    times = []
    for i in range(3):
        print(f"\nAttempt {i+1}/3...")
        t0 = time.time()
        persona = vu_llm.generate_persona()
        t1 = time.time()
        elapsed = t1 - t0
        times.append(elapsed)
        
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Persona: age={persona.get('age')}, style={persona.get('style')}, budget={persona.get('budget')}")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage time: {avg_time:.3f}s")
    
    if avg_time > 5:
        print("[CRITICAL] Average > 5s - VERY SLOW! Check Ollama server performance")
    elif avg_time > 2:
        print("[WARNING] Average > 2s - Consider using fallback mode")
    else:
        print("[OK] Performance is acceptable")
else:
    print("SKIPPED - Ollama not connected")

# Test 4: 페르소나 생성 속도 테스트 (LLM 미사용 - 폴백)
print("\n[Test 4] Persona Generation Speed Test (WITHOUT LLM - Fallback)")
print("-" * 80)

vu_fallback = VirtualUser(None)

times = []
for i in range(3):
    print(f"\nAttempt {i+1}/3...")
    t0 = time.time()
    persona = vu_fallback.generate_persona()
    t1 = time.time()
    elapsed = t1 - t0
    times.append(elapsed)
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Persona: age={persona.get('age')}, style={persona.get('style')}, budget={persona.get('budget')}")

avg_time = sum(times) / len(times)
print(f"\nAverage time: {avg_time:.3f}s")

if avg_time > 0.1:
    print("[WARNING] Fallback mode should be < 0.1s")
else:
    print("[OK] Fallback mode is fast")

# Test 5: API 엔드포인트 테스트
print("\n[Test 5] API Endpoint Test")
print("-" * 80)

if connected:
    import requests
    
    # Test /api/generate
    print("\nTesting /api/generate endpoint...")
    try:
        t0 = time.time()
        r = requests.post(
            f"{ollama_client.base_url}/api/generate",
            json={
                "model": ollama_client.model,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 5}
            },
            timeout=5
        )
        t1 = time.time()
        
        print(f"  Status: {r.status_code}")
        print(f"  Time: {t1-t0:.3f}s")
        
        if r.status_code == 404:
            print("  [WARNING] /api/generate not supported - will use /api/chat fallback")
        elif r.status_code == 200:
            print("  [OK] /api/generate is working")
        else:
            print(f"  [ERROR] Unexpected status code: {r.status_code}")
            
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test /api/chat
    print("\nTesting /api/chat endpoint...")
    try:
        t0 = time.time()
        r = requests.post(
            f"{ollama_client.base_url}/api/chat",
            json={
                "model": ollama_client.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "options": {"num_predict": 5}
            },
            timeout=5
        )
        t1 = time.time()
        
        print(f"  Status: {r.status_code}")
        print(f"  Time: {t1-t0:.3f}s")
        
        if r.status_code == 200:
            print("  [OK] /api/chat is working")
        else:
            print(f"  [ERROR] Unexpected status code: {r.status_code}")
            
    except Exception as e:
        print(f"  [ERROR] {e}")
else:
    print("SKIPPED - Ollama not connected")

# Summary
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

if not connected:
    print("\n[CRITICAL] Ollama is NOT connected!")
    print("Recommendations:")
    print("  1. Check if Ollama is running: ollama serve")
    print("  2. Check if llama3 model is installed: ollama pull llama3")
    print("  3. Verify base URL: http://localhost:11434")
    print("  4. Use fallback mode (--llm 0) for testing")
else:
    print("\n[OK] Ollama is connected")
    print("\nFor best performance:")
    print("  - Use fallback mode (--llm 0) for large-scale simulations")
    print("  - LLM mode is slower but more realistic")
    print("  - Consider reducing num_users if using LLM mode")

print("=" * 80)
