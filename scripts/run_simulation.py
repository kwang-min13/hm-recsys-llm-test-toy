"""
Simulation Runner

가상 유저 시뮬레이션 실행 (optimized with A/B testing)
"""

import sys
from pathlib import Path
import time
import logging
from typing import Dict, Any, List, Optional
import random
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.virtual_user import VirtualUser
from src.simulation.ollama_client import OllamaClient
from src.simulation.ab_test import ABTestSimulator
from src.models.serving import RecommendationService
from src.models.candidate_generation import CandidateGenerator
import duckdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sample_users_fast(num_users: int, seed: int = 42) -> List[str]:
    """
    ORDER BY RANDOM() 제거: 해시 기반으로 빠르게 샘플링
    - 전체 랜덤 정렬 없이, 결정적으로 num_users명을 뽑음
    """
    con = duckdb.connect(":memory:")
    try:
        rows = con.execute(f"""
            SELECT customer_id
            FROM read_parquet('data/features/user_features.parquet')
            ORDER BY abs(hash(customer_id || '{seed}'))
            LIMIT {num_users}
        """).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def rule_based_evaluation(persona: Dict[str, Any], n_recs: int) -> Dict[str, Any]:
    """
    fast 모드용 룰 기반 평가 (초고속)
    - LLM 없이도 그럴듯한 분포를 만들기 위한 간단 규칙
    """
    if n_recs <= 0:
        return {"purchase_count": 0, "acceptance_rate": 0.0}

    budget = persona.get("budget", "medium")

    if budget == "low":
        purchase = random.randint(0, min(2, n_recs))
    elif budget == "high":
        purchase = random.randint(1, min(4, n_recs))

    else:
        purchase = random.randint(0, min(3, n_recs))


    return {
        "purchase_count": purchase,
        "acceptance_rate": purchase / n_recs
    }


def run_ab_test(num_users: int = 20, mode: str = "fast", llm: Optional[int] = None, seed: int = 42):
    """
    A/B 테스트 시뮬레이션 실행
    
    Args:
        num_users: 시뮬레이션할 유저 수
        mode: "fast" (룰 베이스) | "llm" (페르소나 기반)
        llm: LLM 사용 여부 (1=강제 사용, 0=강제 미사용, None=자동)
        seed: 샘플링 seed
    """
    logger.info("=" * 70)
    logger.info(f"A/B 테스트 시뮬레이션 시작 (users={num_users}, mode={mode.upper()}, seed={seed})")
    logger.info("=" * 70)

    start_time = time.time()

    # 1) Ollama 연결 확인
    logger.info("\n[1/5] Ollama 연결 확인 중...")
    ollama_client = OllamaClient()
    conn_ok = ollama_client.check_connection()
    
    # LLM 모드 결정
    if mode == "llm":
        # LLM 모드는 Ollama 필수
        if llm == 0:
            logger.warning("mode=llm이지만 llm=0으로 설정됨. FAST 모드로 전환합니다.")
            simulation_mode = "fast"
            use_llm = False
        elif conn_ok:
            logger.info("✓ Ollama 서버 연결 성공 (LLM 모드)")
            simulation_mode = "llm"
            use_llm = True
        else:
            logger.warning("✗ Ollama 연결 실패. FAST 모드로 전환합니다.")
            simulation_mode = "fast"
            use_llm = False
    else:
        # FAST 모드는 LLM 불필요
        logger.info("✓ FAST 모드 (룰 기반 평가)")
        simulation_mode = "fast"
        use_llm = False

    # 2) 서비스 초기화
    logger.info("\n[2/5] 서비스 초기화 중...")
    rec_service = RecommendationService()
    candidate_gen = CandidateGenerator()
    ab_simulator = ABTestSimulator(
        ollama_client if use_llm else None,
        rec_service,
        candidate_gen,
        mode=simulation_mode
    )

    # 3) 유저 샘플링
    logger.info(f"\n[3/5] 유저 {num_users}명 샘플링 중...")
    user_ids = sample_users_fast(num_users=num_users, seed=seed)
    logger.info(f"샘플링된 유저 수: {len(user_ids)}")

    # 4) A/B 테스트 실행
    logger.info("\n[4/5] A/B 테스트 실행 중...")
    
    results = []
    
    # 그룹 할당 (50:50)
    random.seed(seed)
    
    for i, user_id in enumerate(user_ids, 1):
        # 랜덤 그룹 할당
        group = 'A' if random.random() < 0.5 else 'B'
        
        logger.info(f"\n--- 시뮬레이션 {i}/{num_users} (Group {group}, Mode: {simulation_mode.upper()}) ---")
        
        # 그룹별 시뮬레이션
        try:
            t0 = time.time()
            if group == 'A':
                result = ab_simulator.simulate_group_a(user_id)
            else:
                result = ab_simulator.simulate_group_b(user_id)
            
            t1 = time.time()
            
            logger.info(f"클릭 여부: {result['clicked']}")
            logger.info(f"추천 상품 수: {result['num_items']}개")
            logger.info(f"구매 수: {result.get('purchase_count', 0)}")
            logger.info(f"[timing] sim={t1-t0:.2f}s")
            
            # 결과 저장
            record = {
                'user_id': user_id,
                'group': group,
                'clicked': result['clicked'],
                'purchase_count': result.get('purchase_count', 0),
                'num_items': result['num_items'],
                'mode': result.get('mode', simulation_mode),
                'timestamp': pd.Timestamp.now()
            }
            
            # 사용자 메타데이터(페르소나)가 있으면 추가
            if 'user_metadata' in result:
                meta = result['user_metadata']
                record['persona_age'] = meta.get('age_group')
                record['persona_freq'] = meta.get('purchase_frequency')
                record['persona_price'] = meta.get('avg_price')
                record['persona_recency'] = meta.get('recency')
            
            results.append(record)
            
        except Exception as e:
            logger.exception(f"시뮬레이션 실패 (user={user_id}, group={group}): {e}")
            continue

    # 5) 결과 저장 및 분석
    logger.info("\n[5/5] 결과 저장 및 분석 중...")
    
    if results:
        df = pd.DataFrame(results)
        
        # CSV 저장
        output_path = Path('logs')
        output_path.mkdir(exist_ok=True)
        csv_path = output_path / f'ab_test_results_{simulation_mode}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"결과 저장: {csv_path}")
        
        # 통계 분석
        group_a = df[df['group'] == 'A']
        group_b = df[df['group'] == 'B']
        
        ctr_a = group_a['clicked'].mean() if len(group_a) > 0 else 0
        ctr_b = group_b['clicked'].mean() if len(group_b) > 0 else 0
        
        pur_a = group_a['purchase_count'].mean() if len(group_a) > 0 else 0
        pur_b = group_b['purchase_count'].mean() if len(group_b) > 0 else 0
        
        logger.info("\n" + "=" * 70)
        logger.info(f"A/B 테스트 결과 ({simulation_mode.upper()} 모드)")
        logger.info("=" * 70)
        logger.info(f"총 소요 시간: {time.time() - start_time:.2f}초")
        logger.info(f"성공한 시뮬레이션: {len(results)}/{num_users}")
        logger.info(f"\nGroup A (인기 상품):")
        logger.info(f"  샘플 수: {len(group_a)}")
        logger.info(f"  CTR: {ctr_a:.1%}")
        logger.info(f"  평균 구매 수: {pur_a:.2f}")
        logger.info(f"\nGroup B (ML 추천):")
        logger.info(f"  샘플 수: {len(group_b)}")
        logger.info(f"  CTR: {ctr_b:.1%}")
        logger.info(f"  평균 구매 수: {pur_b:.2f}")
        logger.info(f"\n개선율:")
        logger.info(f"  CTR: {(ctr_b - ctr_a):.1%} ({'↑' if ctr_b > ctr_a else '↓'})")
        logger.info(f"  구매 수: {(pur_b - pur_a):+.2f} ({'↑' if pur_b > pur_a else '↓'})")
        logger.info("=" * 70)
    else:
        logger.warning("시뮬레이션 결과가 없습니다.")
    
    # 정리
    ab_simulator.close()


def run_simulation(num_users: int = 5, mode: str = "fast", llm: Optional[int] = None, seed: int = 42):
    """
    기본 시뮬레이션 실행 (A/B 테스트 아님)
    
    Args:
        num_users: 생성할 가상 유저 수
        mode: "fast" | "full"
        llm: 1이면 LLM 강제 사용, 0이면 강제 미사용, None이면 연결되면 사용
        seed: 샘플링 재현성 seed
    """
    logger.info("=" * 70)
    logger.info(f"가상 유저 시뮬레이션 시작 (users={num_users}, mode={mode}, seed={seed})")
    logger.info("=" * 70)

    start_time = time.time()

    # 1) Ollama 연결 확인
    logger.info("\n[1/4] Ollama 연결 확인 중...")
    ollama_client = OllamaClient()
    conn_ok = ollama_client.check_connection()
    
    if llm == 1:
        use_llm = True
    elif llm == 0:
        use_llm = False
    else:
        use_llm = conn_ok

    if use_llm:
        if conn_ok:
            logger.info("✓ Ollama 서버 연결 성공 (LLM 사용)")
        else:
            logger.warning("⚠ Ollama 연결 실패인데 llm=1로 강제 사용 설정됨. 이후 호출은 실패할 수 있음.")
    else:
        if conn_ok:
            logger.info("✓ Ollama 서버 연결 성공 (하지만 LLM 사용 안 함)")
        else:
            logger.warning("✗ Ollama 서버에 연결할 수 없습니다. 랜덤 페르소나로 진행합니다.")

    # 2) Recommendation Service 초기화
    logger.info("\n[2/4] Recommendation Service 초기화 중...")
    rec_service = RecommendationService()

    # 3) 실제 유저 샘플 추출 (RANDOM 제거)
    logger.info(f"\n[3/4] 실제 유저 {num_users}명 샘플링 중...")
    real_user_ids = sample_users_fast(num_users=num_users, seed=seed)
    logger.info(f"샘플링된 유저 수: {len(real_user_ids)}")

    # 4) 시뮬레이션 실행
    logger.info("\n[4/4] 시뮬레이션 실행 중...")

    results: List[Dict[str, Any]] = []
    vu = VirtualUser(ollama_client if use_llm else None)

    for i, real_user_id in enumerate(real_user_ids, 1):
        logger.info(f"\n--- 시뮬레이션 {i}/{num_users} ---")

        # (A) 페르소나 생성 시간
        t0 = time.time()
        persona = vu.generate_persona()
        t1 = time.time()

        logger.info(
            f"페르소나: {persona.get('age','?')}세 {persona.get('gender','?')}, "
            f"스타일: {persona.get('style', 'N/A')}, 예산: {persona.get('budget', 'N/A')}"
        )

        # (B) 추천 생성 시간
        try:
            recommendations = rec_service.recommend(real_user_id, top_k=10)
        except Exception as e:
            logger.exception(f"추천 생성 중 오류(user={real_user_id}): {e}")
            continue
        t2 = time.time()

        recs = recommendations.get("recommendations") or []
        if not recs:
            logger.warning("추천 결과가 비어있습니다.")
            continue

        # (C) 평가 시간
        try:
            if mode == "fast":
                # ✅ 핵심 최적화: fast 모드에서는 LLM 평가를 끄고 룰 기반 평가
                evaluation = rule_based_evaluation(persona, len(recs))
            else:
                # full 모드에서는 기존처럼 LLM 평가 수행
                evaluation = vu.evaluate_recommendations(recs)
        except Exception as e:
            logger.exception(f"추천 평가 중 오류(user={real_user_id}): {e}")
            continue
        t3 = time.time()

        # 병목 확인 로그
        logger.info(f"[timing] persona={t1-t0:.2f}s rec={t2-t1:.2f}s eval={t3-t2:.2f}s")

        logger.info(f"추천 수: {len(recs)}")
        logger.info(f"구매 예상: {evaluation.get('purchase_count')}")
        logger.info(f"수용률: {evaluation.get('acceptance_rate'):.1%}" if "acceptance_rate" in evaluation else "수용률: N/A")

        results.append({
            "persona": persona,
            "recommendations_count": len(recs),
            "purchase_count": evaluation.get("purchase_count", 0),
            "acceptance_rate": evaluation.get("acceptance_rate", 0.0),
        })

    # 5) 결과 요약
    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 70)
    logger.info("시뮬레이션 완료!")
    logger.info("=" * 70)
    logger.info(f"총 소요 시간: {elapsed:.2f}초")
    logger.info(f"성공한 시뮬레이션: {len(results)}/{num_users}")

    if results:
        avg_purchase = sum(r["purchase_count"] for r in results) / len(results)
        avg_acceptance = sum(r["acceptance_rate"] for r in results) / len(results)

        logger.info("\n평균 지표:")
        logger.info(f"  구매 예상: {avg_purchase:.2f}개")
        logger.info(f"  수용률: {avg_acceptance:.1%}")

    logger.info("=" * 70)

    # 정리
    rec_service.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="가상 유저 시뮬레이션 실행")
    parser.add_argument("--users", type=int, default=5, help="생성할 가상 유저 수")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "llm", "full"], 
                       help="실행 모드 (fast=룰 베이스, llm=페르소나 기반, full=전체 LLM)")
    parser.add_argument("--llm", type=int, default=None, choices=[0, 1], help="LLM 강제 사용(1) / 강제 미사용(0)")
    parser.add_argument("--seed", type=int, default=42, help="유저 샘플링 seed(재현성)")
    parser.add_argument("--ab-test", action="store_true", help="A/B 테스트 모드 활성화")
    args = parser.parse_args()

    if args.ab_test:
        run_ab_test(num_users=args.users, mode=args.mode, llm=args.llm, seed=args.seed)
    else:
        run_simulation(num_users=args.users, mode=args.mode, llm=args.llm, seed=args.seed)
