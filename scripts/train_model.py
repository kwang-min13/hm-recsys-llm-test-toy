"""
Model Training Script

Phase 4: Ranking Model
- Loads data using src.models.dataset
- Splits data into Train/Validation (User-based)
- Trains LightGBM Ranker using src.models.ranker
- Evaluates model using src.models.evaluation
- Saves model to models/artifacts/purchase_ranker.pkl
"""

import sys
from pathlib import Path
import logging
import numpy as np
import polars as pl
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.dataset import create_ranking_dataset
from src.models.ranker import PurchaseRanker
from src.models.evaluation import evaluate_ranking
from src.models.user_segmentation import UserSegmenter, UserSegment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_data(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    group: List[int],
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[pl.DataFrame, pl.DataFrame, List[int], pl.DataFrame, pl.DataFrame, List[int]]:
    """
    User-based random split
    """
    logger.info(f"Splitting data (validation ratio: {val_ratio})...")
    
    n_users = len(group)
    n_val = int(n_users * val_ratio)
    
    # User indices
    indices = np.arange(n_users)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    val_indices = set(indices[:n_val])
    
    # Build boolean mask for rows
    # This loop might be slow for millions of users, but efficient enough for 10k-100k
    train_mask = []
    val_mask = []
    
    # Also separate groups
    train_group = []
    val_group = []
    
    current_idx = 0
    # Group is list of row counts per user
    for i, count in enumerate(group):
        is_val = i in val_indices
        if is_val:
            val_group.append(count)
            val_mask.extend([True] * count)
            train_mask.extend([False] * count)
        else:
            train_group.append(count)
            val_mask.extend([False] * count)
            train_mask.extend([True] * count)
            
    # Polars filtering
    X_train = features.filter(pl.Series(train_mask))
    y_train = labels.filter(pl.Series(train_mask))
    
    X_val = features.filter(pl.Series(val_mask))
    y_val = labels.filter(pl.Series(val_mask))
    
    logger.info(f"Train: {X_train.height} rows, {len(train_group)} users")
    logger.info(f"Valid: {X_val.height} rows, {len(val_group)} users")
    
    return X_train, y_train, train_group, X_val, y_val, val_group


def convert_group_to_tuples(group: List[int]) -> List[Tuple[int, int]]:
    """Convert count-based group to (start, end) tuples for evaluation"""
    tuples = []
    start_idx = 0
    for count in group:
        end_idx = start_idx + count
        tuples.append((start_idx, end_idx))
        start_idx = end_idx
    return tuples


def profile(df: pl.DataFrame, name: str):
    logger.info(f"[{name}] shape={df.shape}")
    for c in df.columns:
        s = df.select([
            pl.col(c).n_unique().alias("n_unique"),
            pl.col(c).null_count().alias("nulls"),
            pl.col(c).min().alias("min"),
            pl.col(c).max().alias("max"),
        ]).row(0)
        logger.info(f"[{name}] {c:28s} n_unique={s[0]:6d} nulls={s[1]:6d} min={s[2]} max={s[3]}")

def main():
    logger.info("="*60)
    logger.info("Start Model Training Pipeline")
    logger.info("="*60)

    # 1. Dataset Creation
    # Check if we should use full data or smaller sample
    # For now, let's use a reasonable sample size for quick training but large enough for meaningful results
    X, y, group = create_ranking_dataset(
        sample_users= 10000, # 샘플 유저
        pos_per_user=3, # 긍정 샘플
        negative_per_user= 5, # 부정 샘플
        window_days=60, # 최근성 반영
        popularity_pool=1000, # 인기 아이템 풀
        seed=42 
    )

    # 2. Train/Val Split
    X_train, y_train, group_train, X_val, y_val, group_val = split_data(X, y, group)

    # 3. Model Training
    ranker = PurchaseRanker()
    
    # Custom params if needed
    params = {
        "objective": "lambdarank", #목적 함수 지정
        "metric": "ndcg", #모델 성능 평가
        "eval_at": [10, 20], # 상위 컷
        "boosting_type": "gbdt", # 부스팅 타입
        "learning_rate": 0.01, # 학습률
        "num_leaves": 26,  # 트리 복잡도
        "feature_fraction": 1.0, # 서브 샘플링 (과적합을 줄인다.)
        "bagging_fraction": 0.9, # 각 반복에서 데이터 샘플링 과적합 방지 및 분산 줄이기
        "bagging_freq": 2, # 배깅 반복 수행
        "max_depth": -1, # 트리 깊이 제한 (-1 제한x)
        "seed": 42,
        'feature_pre_filter': False, 
        'lambda_l1': 0.0, 
        'lambda_l2': 0.0, 
        'min_child_samples': 20
    }

    ranker.train(
        X_train, y_train, group_train,
        X_val, y_val, group_val,
        params=params,
        num_boost_round=500,
        early_stopping_rounds=30
    )

    # 4. Evaluation
    logger.info("\nEvaluating on Validation Set...")
    val_groups_tuples = convert_group_to_tuples(group_val)
    val_preds = ranker.predict(X_val)
    val_true = y_val.to_numpy().ravel()
    
    metrics = evaluate_ranking(val_true, val_preds, val_groups_tuples, k_values=[5, 10, 20])
    
    logger.info("-" * 40)
    logger.info("Final Validation Metrics:")
    for k, v in metrics.items():
        logger.info(f"{k:<20}: {v:.4f}")
    logger.info("-" * 40)
    
    # Feature Importance
    logger.info("\nFeature Importance:")
    imp = ranker.get_feature_importance()
    for name, gain in sorted(imp.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {name:<20}: {gain:.4f}")

    # 5. Save Model
    save_path = "models/artifacts/purchase_ranker.pkl"
    ranker.save(save_path)
    logger.info(f"\nModel saved to {save_path}")
    logger.info(f"Compatible with loading via: PurchaseRanker.load('{save_path}')")
    logger.info("="*60)

if __name__ == "__main__":
    main()
