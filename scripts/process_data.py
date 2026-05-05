"""
Data Processing Pipeline

User Features와 Item Features를 한번에 생성하는 통합 스크립트
"""

import sys
from pathlib import Path
import logging

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.user_features import UserFeatureGenerator
from src.data.item_features import ItemFeatureGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """전체 Feature 생성 파이프라인"""
    logger.info("=" * 70)
    logger.info("Local_Helix Data Processing Pipeline")
    logger.info("=" * 70)
    
    # 1. User Features 생성
    logger.info("\n[Step 1/2] User Features 생성")
    logger.info("-" * 70)
    
    user_gen = UserFeatureGenerator()
    try:
        user_output = user_gen.create_user_features(
            transactions_path='data/transactions_train.csv',
            output_path='data/features/user_features.parquet',
            lookback_days=28
        )
        logger.info(f"✓ User Features 저장 완료: {user_output}")
    except Exception as e:
        logger.error(f"✗ User Features 생성 실패: {e}")
        return 1
    finally:
        user_gen.close()
    
    # 2. Item Features 생성
    logger.info("\n[Step 2/2] Item Features 생성")
    logger.info("-" * 70)
    
    item_gen = ItemFeatureGenerator()
    try:
        item_output = item_gen.create_item_features(
            transactions_path='data/transactions_train.csv',
            articles_path='data/articles.csv',
            output_path='data/features/item_features.parquet',
            lookback_days=7
        )
        logger.info(f"✓ Item Features 저장 완료: {item_output}")
    except Exception as e:
        logger.error(f"✗ Item Features 생성 실패: {e}")
        return 1
    finally:
        item_gen.close()
    
    # 완료
    logger.info("\n" + "=" * 70)
    logger.info("✓ 모든 Feature 생성 완료!")
    logger.info("=" * 70)
    logger.info("\n다음 단계:")
    logger.info("  2. 모델 학습: python scripts/train_model.py")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
