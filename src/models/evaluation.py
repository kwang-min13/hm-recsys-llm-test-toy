"""
Model Evaluation Module

이 모듈은 추천 모델의 성능을 평가하는 다양한 메트릭을 제공합니다.
- NDCG@K: Normalized Discounted Cumulative Gain
- Hit Rate@K: Top K 추천 중 실제 구매가 포함된 비율
- Precision@K, Recall@K: 정밀도와 재현율
"""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dcg_at_k(relevance_scores: np.ndarray, k: int) -> float:
    """
    Discounted Cumulative Gain at K
    
    Args:
        relevance_scores: 관련성 점수 배열 (이미 정렬된 상태)
        k: 상위 K개
        
    Returns:
        DCG@K 값
    """
    relevance_scores = np.asarray(relevance_scores)[:k]
    if relevance_scores.size == 0:
        return 0.0
    
    # DCG = sum(rel_i / log2(i+2)) for i in range(k)
    discounts = np.log2(np.arange(2, relevance_scores.size + 2))
    return np.sum(relevance_scores / discounts)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K
    
    Args:
        y_true: 실제 레이블 (0 or 1)
        y_pred: 예측 점수
        k: 상위 K개
        
    Returns:
        NDCG@K 값 (0~1)
    """
    # 예측 점수로 정렬
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    
    # DCG 계산
    dcg = dcg_at_k(y_true_sorted, k)
    
    # IDCG (Ideal DCG) 계산
    ideal_order = np.argsort(y_true)[::-1]
    y_true_ideal = y_true[ideal_order]
    idcg = dcg_at_k(y_true_ideal, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def hit_rate_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Hit Rate at K: Top K 추천 중 실제 구매가 하나라도 포함된 비율
    
    Args:
        y_true: 실제 레이블 (0 or 1)
        y_pred: 예측 점수
        k: 상위 K개
        
    Returns:
        Hit Rate (0 or 1 for single user)
    """
    # 예측 점수로 정렬하여 Top K 추출
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    top_k_labels = y_true[top_k_indices]
    
    # Top K 중 하나라도 1이 있으면 hit
    return 1.0 if np.sum(top_k_labels) > 0 else 0.0


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Precision at K: Top K 추천 중 실제 구매 비율
    
    Args:
        y_true: 실제 레이블 (0 or 1)
        y_pred: 예측 점수
        k: 상위 K개
        
    Returns:
        Precision@K (0~1)
    """
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    top_k_labels = y_true[top_k_indices]
    
    return np.sum(top_k_labels) / k


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Recall at K: 전체 실제 구매 중 Top K에 포함된 비율
    
    Args:
        y_true: 실제 레이블 (0 or 1)
        y_pred: 예측 점수
        k: 상위 K개
        
    Returns:
        Recall@K (0~1)
    """
    total_relevant = np.sum(y_true)
    if total_relevant == 0:
        return 0.0
    
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    top_k_labels = y_true[top_k_indices]
    
    return np.sum(top_k_labels) / total_relevant


def evaluate_ranking(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    user_groups: List[Tuple[int, int]],
    k_values: List[int] = [5, 10, 20, 50]
) -> Dict[str, float]:
    """
    유저별 랭킹 평가 메트릭 계산
    
    Args:
        y_true: 전체 실제 레이블 배열
        y_pred: 전체 예측 점수 배열
        user_groups: 유저별 (start_idx, end_idx) 리스트
        k_values: 평가할 K 값들
        
    Returns:
        메트릭 딕셔너리 (평균값)
    """
    metrics = defaultdict(list)
    
    for start_idx, end_idx in user_groups:
        user_y_true = y_true[start_idx:end_idx]
        user_y_pred = y_pred[start_idx:end_idx]
        
        # 각 K에 대해 메트릭 계산
        for k in k_values:
            metrics[f'ndcg@{k}'].append(ndcg_at_k(user_y_true, user_y_pred, k))
            metrics[f'hit_rate@{k}'].append(hit_rate_at_k(user_y_true, user_y_pred, k))
            metrics[f'precision@{k}'].append(precision_at_k(user_y_true, user_y_pred, k))
            metrics[f'recall@{k}'].append(recall_at_k(user_y_true, user_y_pred, k))
    
    # 평균 계산
    avg_metrics = {
        metric_name: np.mean(values)
        for metric_name, values in metrics.items()
    }
    
    return avg_metrics


def evaluate_model(
    model,
    X_val: pl.DataFrame,
    y_val: pl.DataFrame,
    user_groups: List[Tuple[int, int]],
    k_values: List[int] = [5, 10, 20, 50]
) -> Dict[str, float]:
    """
    모델 평가 (전체 파이프라인)
    
    Args:
        model: 학습된 모델 (predict 메서드 필요)
        X_val: 검증 Feature DataFrame
        y_val: 검증 레이블 DataFrame
        user_groups: 유저별 (start_idx, end_idx) 리스트
        k_values: 평가할 K 값들
        
    Returns:
        평가 메트릭 딕셔너리
    """
    logger.info("모델 평가 시작...")
    
    # 예측
    y_pred = model.predict(X_val)
    y_true = y_val.to_numpy().ravel()
    
    # 평가
    metrics = evaluate_ranking(y_true, y_pred, user_groups, k_values)
    
    # 로깅
    logger.info("=" * 60)
    logger.info("모델 평가 결과:")
    for metric_name, value in sorted(metrics.items()):
        logger.info(f"  {metric_name}: {value:.4f}")
    logger.info("=" * 60)
    
    return metrics


def create_user_groups_from_polars(
    df: pl.DataFrame,
    user_col: str = 'customer_id'
) -> List[Tuple[int, int]]:
    """
    Polars DataFrame에서 유저별 그룹 인덱스 생성
    
    Args:
        df: 데이터프레임 (user_col로 정렬되어 있어야 함)
        user_col: 유저 ID 컬럼명
        
    Returns:
        유저별 (start_idx, end_idx) 리스트
    """
    # 유저별 카운트
    user_counts = df.group_by(user_col).agg(pl.count().alias('count'))
    user_counts = user_counts.sort(user_col)
    
    groups = []
    start_idx = 0
    for count in user_counts['count'].to_list():
        end_idx = start_idx + count
        groups.append((start_idx, end_idx))
        start_idx = end_idx
    
    return groups


def main():
    """테스트 함수"""
    # 간단한 테스트
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    
    print("Test Evaluation Metrics:")
    print(f"NDCG@3: {ndcg_at_k(y_true, y_pred, 3):.4f}")
    print(f"Hit Rate@3: {hit_rate_at_k(y_true, y_pred, 3):.4f}")
    print(f"Precision@3: {precision_at_k(y_true, y_pred, 3):.4f}")
    print(f"Recall@3: {recall_at_k(y_true, y_pred, 3):.4f}")


if __name__ == "__main__":
    main()
