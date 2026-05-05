"""
Validation Utilities

이 모듈은 데이터와 모델의 유효성을 검증하는 함수들을 제공합니다.
"""

import polars as pl
import numpy as np
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """검증 실패 시 발생하는 예외"""
    pass


def validate_dataframe(
    df: pl.DataFrame,
    required_columns: List[str],
    min_rows: int = 1,
    check_nulls: bool = True
) -> bool:
    """
    DataFrame 유효성 검증
    
    Args:
        df: 검증할 DataFrame
        required_columns: 필수 컬럼 리스트
        min_rows: 최소 행 수
        check_nulls: NULL 체크 여부
        
    Returns:
        검증 성공 여부
        
    Raises:
        ValidationError: 검증 실패 시
    """
    # 행 수 체크
    if len(df) < min_rows:
        raise ValidationError(f"행 수가 부족합니다: {len(df)} < {min_rows}")
    
    # 필수 컬럼 체크
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValidationError(f"필수 컬럼이 없습니다: {missing_columns}")
    
    # NULL 체크
    if check_nulls:
        for col in required_columns:
            null_count = df[col].null_count()
            if null_count > 0:
                logger.warning(f"컬럼 '{col}'에 {null_count}개의 NULL 값이 있습니다.")
    
    return True


def validate_feature_ranges(
    df: pl.DataFrame,
    feature_ranges: Dict[str, tuple]
) -> bool:
    """
    Feature 값 범위 검증
    
    Args:
        df: 검증할 DataFrame
        feature_ranges: {컬럼명: (min, max)} 딕셔너리
        
    Returns:
        검증 성공 여부
        
    Raises:
        ValidationError: 범위를 벗어난 값이 있을 때
    """
    for col, (min_val, max_val) in feature_ranges.items():
        if col not in df.columns:
            logger.warning(f"컬럼 '{col}'이 DataFrame에 없습니다.")
            continue
        
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min < min_val or col_max > max_val:
            raise ValidationError(
                f"컬럼 '{col}'의 값이 범위를 벗어났습니다: "
                f"[{col_min}, {col_max}] not in [{min_val}, {max_val}]"
            )
    
    return True


def validate_user_features(df: pl.DataFrame) -> bool:
    """
    User Feature DataFrame 검증
    
    Args:
        df: User Feature DataFrame
        
    Returns:
        검증 성공 여부
    """
    required_columns = [
        'customer_id',
        'avg_purchase_hour',
        'purchase_count',
        'recency'
    ]
    
    validate_dataframe(df, required_columns, min_rows=1)
    
    # 범위 검증
    feature_ranges = {
        'avg_purchase_hour': (0, 24),
        'purchase_count': (0, float('inf')),
        'recency': (0, float('inf'))
    }
    
    validate_feature_ranges(df, feature_ranges)
    
    logger.info(f"✓ User Feature 검증 완료: {len(df):,}개 유저")
    return True


def validate_item_features(df: pl.DataFrame) -> bool:
    """
    Item Feature DataFrame 검증
    
    Args:
        df: Item Feature DataFrame
        
    Returns:
        검증 성공 여부
    """
    required_columns = [
        'article_id',
        'popularity_rank',
        'sales_count'
    ]
    
    validate_dataframe(df, required_columns, min_rows=1)
    
    # 범위 검증
    feature_ranges = {
        'popularity_rank': (1, float('inf')),
        'sales_count': (0, float('inf'))
    }
    
    validate_feature_ranges(df, feature_ranges)
    
    logger.info(f"✓ Item Feature 검증 완료: {len(df):,}개 상품")
    return True


def validate_predictions(
    predictions: np.ndarray,
    expected_length: int,
    check_range: bool = True
) -> bool:
    """
    모델 예측 결과 검증
    
    Args:
        predictions: 예측 배열
        expected_length: 예상 길이
        check_range: 값 범위 체크 여부 (0~1)
        
    Returns:
        검증 성공 여부
        
    Raises:
        ValidationError: 검증 실패 시
    """
    # 길이 체크
    if len(predictions) != expected_length:
        raise ValidationError(
            f"예측 길이가 일치하지 않습니다: {len(predictions)} != {expected_length}"
        )
    
    # NaN/Inf 체크
    if np.isnan(predictions).any():
        raise ValidationError("예측에 NaN 값이 포함되어 있습니다.")
    
    if np.isinf(predictions).any():
        raise ValidationError("예측에 Inf 값이 포함되어 있습니다.")
    
    # 범위 체크 (선택)
    if check_range:
        if predictions.min() < 0 or predictions.max() > 1:
            logger.warning(
                f"예측 값이 [0, 1] 범위를 벗어났습니다: "
                f"[{predictions.min():.4f}, {predictions.max():.4f}]"
            )
    
    logger.info(f"✓ 예측 검증 완료: {len(predictions):,}개")
    return True


def validate_model_output(
    recommendations: List[str],
    min_items: int = 1,
    max_items: int = 100
) -> bool:
    """
    추천 결과 검증
    
    Args:
        recommendations: 추천 상품 ID 리스트
        min_items: 최소 추천 개수
        max_items: 최대 추천 개수
        
    Returns:
        검증 성공 여부
        
    Raises:
        ValidationError: 검증 실패 시
    """
    # 개수 체크
    if len(recommendations) < min_items:
        raise ValidationError(
            f"추천 개수가 부족합니다: {len(recommendations)} < {min_items}"
        )
    
    if len(recommendations) > max_items:
        raise ValidationError(
            f"추천 개수가 너무 많습니다: {len(recommendations)} > {max_items}"
        )
    
    # 중복 체크
    if len(recommendations) != len(set(recommendations)):
        logger.warning("추천 결과에 중복이 있습니다.")
    
    # 빈 값 체크
    if any(not item for item in recommendations):
        raise ValidationError("추천 결과에 빈 값이 있습니다.")
    
    logger.info(f"✓ 추천 결과 검증 완료: {len(recommendations)}개 상품")
    return True


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    설정 딕셔너리 검증
    
    Args:
        config: 설정 딕셔너리
        required_keys: 필수 키 리스트
        
    Returns:
        검증 성공 여부
        
    Raises:
        ValidationError: 필수 키가 없을 때
    """
    missing_keys = set(required_keys) - set(config.keys())
    if missing_keys:
        raise ValidationError(f"필수 설정이 없습니다: {missing_keys}")
    
    logger.info("✓ 설정 검증 완료")
    return True


def main():
    """테스트"""
    # DataFrame 검증 테스트
    df = pl.DataFrame({
        'customer_id': ['user1', 'user2'],
        'avg_purchase_hour': [14.5, 19.2],
        'purchase_count': [10, 5],
        'recency': [3, 7]
    })
    
    try:
        validate_user_features(df)
        print("✓ User Feature 검증 성공")
    except ValidationError as e:
        print(f"✗ 검증 실패: {e}")
    
    # 예측 검증 테스트
    predictions = np.array([0.8, 0.6, 0.3, 0.9])
    try:
        validate_predictions(predictions, expected_length=4)
        print("✓ 예측 검증 성공")
    except ValidationError as e:
        print(f"✗ 검증 실패: {e}")


if __name__ == "__main__":
    main()
