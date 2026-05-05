import lightgbm as lgb
import optuna.integration.lightgbm as lgb_optuna  # 자동 튜너
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

def to_lgb_inputs(X_pl: pl.DataFrame, y_pl: pl.DataFrame):
    # X: polars -> numpy(float32) or pandas(float32)
    X_np = X_pl.to_numpy()
    X_np = X_np.astype(np.float32, copy=False)

    # y: polars(DataFrame/Series) -> 1d numpy
    y_np = y_pl.to_numpy().ravel()
    y_np = y_np.astype(np.float32, copy=False)

    return X_np, y_np
    
def to_label_1d(y_pl: pl.DataFrame | pl.Series):
    if isinstance(y_pl, pl.DataFrame):
        assert y_pl.width == 1, f"label columns must be 1, got {y_pl.width}"
        y_np = y_pl.select(pl.all()).to_numpy().reshape(-1)
    else:
        y_np = y_pl.to_numpy()
    return y_np.astype(np.float32, copy=False)

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
    
    Xtr, ytr = to_lgb_inputs(X_train, y_train)
    Xva, yva = to_lgb_inputs(X_val, y_val)
    
    train_data = lgb.Dataset(Xtr, label=ytr, group=group_train)
    valid_data = lgb.Dataset(Xva, label=yva, group=group_val)

    params_init = {
        "objective": "lambdarank", #목적 함수 지정
        "metric": "ndcg", #모델 성능 평가
        "eval_at": [10, 20], # 상위 컷
        "boosting_type": "gbdt", # 부스팅 타입
        "learning_rate": 0.01, # 학습률
        "num_leaves": 31,  # 트리 복잡도
        "feature_fraction": 0.8, # 서브 샘플링 (과적합을 줄인다.)
        "bagging_fraction": 0.7, # 각 반복에서 데이터 샘플링 과적합 방지 및 분산 줄이기
        "bagging_freq": 1, # 배깅 반복 수행
        "max_depth": -1, # 트리 깊이 제한 (-1 제한x)
        "seed": 42
    }

    # Optuna 튜너 실행 (CV가 아닌 Hold-out validation 사용)
    # LightGBMTunerCV는 내부적으로 CV를 수행하므로, split_data로 나눈 valid_data를 활용하려면 LightGBMTuner를 사용해야 합니다.
    tuner = lgb_optuna.LightGBMTuner(
        params_init,
        train_data,
        valid_sets=[valid_data],
        time_budget=1200,  # 1시간 예산
    )
    tuner.run()

    logger.info(f"Best params: {tuner.best_params}")
    logger.info(f"Best score: {tuner.best_score}")


if __name__ == "__main__":
    main()