"""
Candidate Generation Test Script

후보군 생성 로직을 테스트하고 품질을 검증합니다.
"""

import sys
from pathlib import Path
import logging

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.candidate_generation import CandidateGenerator
import duckdb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_candidates():
    """후보군 생성 테스트"""
    logger.info("=" * 70)
    logger.info("Candidate Generation Test")
    logger.info("=" * 70)
    
    # 후보군 생성기 초기화
    gen = CandidateGenerator(
        db_path="local_helix.db",
        transactions_path="data/transactions_train.csv",
        item_features_path="data/features/item_features.parquet",
        cf_window_days=28,
        materialize_transactions=True
    )
    
    try:
        # 샘플 유저 선택
        con = gen.connect()
        sample_users = con.execute("""
            SELECT customer_id
            FROM read_parquet('data/features/user_features.parquet')
            ORDER BY purchase_count DESC
            LIMIT 5
        """).fetchall()
        
        logger.info(f"\n테스트 유저 수: {len(sample_users)}")
        
        for idx, (user_id,) in enumerate(sample_users, 1):
            logger.info(f"\n[Test {idx}/5] User: {user_id}")
            logger.info("-" * 70)
            
            # 1. Popularity 후보군
            pop_candidates = gen.generate_popularity_candidates(top_k=50)
            logger.info(f"✓ Popularity candidates: {len(pop_candidates)}")
            logger.info(f"  Top 5: {pop_candidates[:5]}")
            
            # 2. CF 후보군
            cf_candidates = gen.generate_cf_scored_item2item(
                user_id=user_id,
                top_k=50
            )
            logger.info(f"✓ CF candidates: {len(cf_candidates)}")
            if cf_candidates:
                logger.info(f"  Top 5: {[c.item_id for c in cf_candidates[:5]]}")
                logger.info(f"  Scores: {[f'{c.score:.4f}' for c in cf_candidates[:5]]}")
            
            # 3. 병합된 후보군
            merged = gen.merge_candidates(
                user_id=user_id,
                total_k=100
            )
            logger.info(f"✓ Merged candidates: {len(merged)}")
            logger.info(f"  Top 10: {merged[:10]}")
            
            # 품질 검증
            if len(merged) < 50:
                logger.warning(f"  ⚠ 후보군이 50개 미만입니다: {len(merged)}")
            
            # 중복 체크
            if len(merged) != len(set(merged)):
                logger.error(f"  ✗ 중복된 후보가 있습니다!")
            else:
                logger.info(f"  ✓ 중복 없음")
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ 후보군 생성 테스트 완료!")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        gen.close()


if __name__ == "__main__":
    sys.exit(test_candidates())
