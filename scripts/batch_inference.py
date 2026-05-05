"""
Batch Inference Script

전체 유저에 대한 배치 추론 실행
"""

import sys
from pathlib import Path
import time
import logging
from tqdm import tqdm
import duckdb

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.serving import RecommendationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(sample_size: int = 100):
    """
    배치 추론 실행
    
    Args:
        sample_size: 처리할 유저 수
    """
    logger.info("=" * 70)
    logger.info("배치 추론 시작")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    # 1. Recommendation Service 초기화
    logger.info("\n[1/3] Recommendation Service 초기화 중...")
    service = RecommendationService()
    
    # 2. 샘플 유저 추출 (최적화: hash 기반 샘플링)
    logger.info(f"\n[2/3] 샘플 유저 {sample_size}명 추출 중...")
    con = duckdb.connect(':memory:')
    sample_users = con.execute(f"""
        SELECT customer_id 
        FROM read_parquet('data/features/user_features.parquet')
        ORDER BY abs(hash(customer_id))
        LIMIT {sample_size}
    """).fetchall()
    con.close()
    
    logger.info(f"추출된 유저 수: {len(sample_users)}")
    
    # 3. 배치 추론
    logger.info(f"\n[3/3] 추천 생성 중...")
    
    results = []
    failed = []
    
    for user_row in tqdm(sample_users, desc="추천 생성"):
        user_id = user_row[0]
        
        try:
            result = service.recommend(user_id, top_k=10)
            
            if result['recommendations']:
                results.append(result)
            else:
                failed.append(user_id)
                
        except Exception as e:
            logger.error(f"유저 {user_id} 추천 실패: {str(e)}")
            failed.append(user_id)
    
    # 4. 결과 저장 (간단한 요약)
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("배치 추론 완료!")
    logger.info("=" * 70)
    logger.info(f"총 소요 시간: {elapsed:.2f}초")
    logger.info(f"성공: {len(results)}명")
    logger.info(f"실패: {len(failed)}명")
    logger.info(f"평균 처리 시간: {elapsed/len(sample_users)*1000:.2f}ms/유저")
    
    # 샘플 결과 출력
    if results:
        logger.info(f"\n샘플 추천 결과 (처음 3명):")
        for i, result in enumerate(results[:3], 1):
            logger.info(f"\n  유저 {i}: {result['user_id']}")
            logger.info(f"    추천 상품: {result['recommendations'][:5]}")
    
    logger.info("=" * 70)
    
    # 정리
    service.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='배치 추론 실행')
    parser.add_argument('--sample-size', type=int, default=100, help='처리할 유저 수')
    args = parser.parse_args()
    
    main(sample_size=args.sample_size)
