"""
Persona Generation Module

이 모듈은 LLM 기반 A/B 테스트를 위한 가상 유저 페르소나를 생성합니다.
실제 유저 데이터를 기반으로 현실적인 쇼핑 성향을 가진 페르소나를 만듭니다.
"""

from __future__ import annotations

import duckdb
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserMetadata:
    """유저 메타데이터 스키마"""
    user_id: str
    age_group: str  # '20s', '30s', '40s', '50s+'
    preferred_category: Optional[str]
    recent_purchases: List[str]
    purchase_frequency: str  # 'high', 'medium', 'low'
    avg_price: float
    recency: int  # 마지막 구매로부터 경과일


# 페르소나 특성 정의
AGE_TRAITS = {
    '20s': '트렌디하고 새로운 스타일을 추구하는',
    '30s': '실용적이면서도 품질을 중시하는',
    '40s': '클래식하고 안정적인 스타일을 선호하는',
    '50s+': '편안함과 품질을 최우선으로 하는',
}

FREQUENCY_TRAITS = {
    'high': '패션에 관심이 많아 자주 쇼핑하는',
    'medium': '필요할 때 계획적으로 구매하는',
    'low': '신중하게 선택하여 가끔 구매하는',
}

PRICE_SENSITIVITY = {
    'budget': '가성비를 중요하게 생각하는',
    'moderate': '적절한 가격대의 상품을 선호하는',
    'premium': '품질을 위해 프리미엄 가격도 기꺼이 지불하는',
}


def classify_price_sensitivity(avg_price: float) -> str:
    """
    평균 구매 가격으로 가격 민감도 분류
    
    Args:
        avg_price: 평균 구매 가격
        
    Returns:
        가격 민감도 ('budget', 'moderate', 'premium')
    """
    if avg_price < 0.02:  # H&M 데이터는 가격이 매우 작은 단위
        return 'budget'
    elif avg_price < 0.04:
        return 'moderate'
    else:
        return 'premium'


def create_persona(user_metadata: UserMetadata) -> str:
    """
    유저 메타데이터 기반 페르소나 프롬프트 생성
    
    Args:
        user_metadata: 유저 메타데이터
        
    Returns:
        LLM에 전달할 페르소나 프롬프트
    """
    age_trait = AGE_TRAITS.get(user_metadata.age_group, '일반적인')
    frequency_trait = FREQUENCY_TRAITS.get(user_metadata.purchase_frequency, '고객')
    price_trait = PRICE_SENSITIVITY.get(
        classify_price_sensitivity(user_metadata.avg_price),
        '적절한 가격대의 상품을 선호하는'
    )
    
    # 최근 구매 상품 (최대 3개)
    recent_items_str = ', '.join(user_metadata.recent_purchases[:3]) if user_metadata.recent_purchases else '없음'
    
    # 최근성 해석
    if user_metadata.recency <= 7:
        recency_desc = '최근 일주일 내에 구매한'
    elif user_metadata.recency <= 30:
        recency_desc = '최근 한 달 내에 구매한'
    else:
        recency_desc = f'{user_metadata.recency}일 전에 마지막으로 구매한'
    
    template = f"""당신은 다음과 같은 특성을 가진 H&M 고객입니다:

**기본 정보**:
- 연령대: {user_metadata.age_group}
- 쇼핑 성향: {age_trait} {frequency_trait} 고객
- 가격 선호도: {price_trait}

**쇼핑 패턴**:
- 구매 빈도: {user_metadata.purchase_frequency}
- 마지막 구매: {recency_desc}
- 평균 구매 가격: {user_metadata.avg_price:.4f}

**최근 관심사**:
- 주로 구매하는 카테고리: {user_metadata.preferred_category or '다양함'}
- 최근 구매 상품: {recent_items_str}

당신의 쇼핑 성향과 선호도를 바탕으로 솔직하게 행동해주세요.
푸시 알림을 받았을 때, 상품이 당신의 취향에 맞는지, 시간대가 적절한지 고려하여 판단하세요."""

    return template


def create_simple_persona(
    age_group: str = '30s',
    purchase_frequency: str = 'medium',
    avg_price: float = 0.03
) -> str:
    """
    간단한 페르소나 생성 (메타데이터 없이)
    
    Args:
        age_group: 연령대
        purchase_frequency: 구매 빈도
        avg_price: 평균 가격
        
    Returns:
        페르소나 프롬프트
    """
    metadata = UserMetadata(
        user_id='test',
        age_group=age_group,
        preferred_category=None,
        recent_purchases=[],
        purchase_frequency=purchase_frequency,
        avg_price=avg_price,
        recency=7
    )
    return create_persona(metadata)


def load_user_metadata(
    con: duckdb.DuckDBPyConnection,
    user_id: str
) -> Optional[UserMetadata]:
    """
    DuckDB에서 유저 메타데이터 로드
    
    Args:
        con: DuckDB 연결
        user_id: 유저 ID
        
    Returns:
        UserMetadata 객체 또는 None
    """
    query = """
    SELECT 
        customer_id,
        CASE 
            WHEN abs(hash(customer_id)) % 100 < 25 THEN '20s'
            WHEN abs(hash(customer_id)) % 100 < 50 THEN '30s'
            WHEN abs(hash(customer_id)) % 100 < 75 THEN '40s'
            ELSE '50s+'
        END as age_group,
        NULL as preferred_category,
        '[]' as recent_purchases,
        COALESCE(purchase_frequency, 'medium') as purchase_frequency,
        COALESCE(avg_price, 0.03) as avg_price,
        COALESCE(recency, 7) as recency
    FROM filedb.user_features
    WHERE customer_id = ?
    LIMIT 1
    """
    
    try:
        result = con.execute(query, [user_id]).fetchone()
        
        if result is None:
            logger.warning(f"유저 {user_id}를 찾을 수 없습니다.")
            return None
        
        # recent_purchases는 문자열로 저장되어 있으므로 파싱 필요
        # 여기서는 간단히 빈 리스트로 처리
        recent_purchases = []
        
        return UserMetadata(
            user_id=str(result[0]),
            age_group=result[1],
            preferred_category=result[2],
            recent_purchases=recent_purchases,
            purchase_frequency=result[4],
            avg_price=float(result[5]),
            recency=int(result[6])
        )
    
    except Exception as e:
        logger.error(f"유저 메타데이터 로드 실패: {str(e)}")
        return None


def load_user_metadata_batch(
    con: duckdb.DuckDBPyConnection,
    user_ids: List[str]
) -> Dict[str, UserMetadata]:
    """
    여러 유저의 메타데이터를 한 번에 로드
    
    Args:
        con: DuckDB 연결
        user_ids: 유저 ID 리스트
        
    Returns:
        {user_id: UserMetadata} 딕셔너리
    """
    if not user_ids:
        return {}
    
    # IN 절을 위한 플레이스홀더
    placeholders = ','.join(['?' for _ in user_ids])
    
    query = f"""
    SELECT 
        customer_id,
        CASE 
            WHEN abs(hash(customer_id)) % 100 < 25 THEN '20s'
            WHEN abs(hash(customer_id)) % 100 < 50 THEN '30s'
            WHEN abs(hash(customer_id)) % 100 < 75 THEN '40s'
            ELSE '50s+'
        END as age_group,
        NULL as preferred_category,
        COALESCE(purchase_frequency, 'medium') as purchase_frequency,
        COALESCE(avg_price, 0.03) as avg_price,
        COALESCE(recency, 7) as recency
    FROM filedb.user_features
    WHERE customer_id IN ({placeholders})
    """
    
    try:
        results = con.execute(query, user_ids).fetchall()
        
        metadata_dict = {}
        for row in results:
            metadata = UserMetadata(
                user_id=str(row[0]),
                age_group=row[1],
                preferred_category=row[2],
                recent_purchases=[],
                purchase_frequency=row[3],
                avg_price=float(row[4]),
                recency=int(row[5])
            )
            metadata_dict[str(row[0])] = metadata
        
        return metadata_dict
    
    except Exception as e:
        logger.error(f"배치 메타데이터 로드 실패: {str(e)}")
        return {}


def main():
    """테스트 함수"""
    # 테스트 메타데이터
    test_metadata = UserMetadata(
        user_id='test_user_123',
        age_group='30s',
        preferred_category='Tops',
        recent_purchases=['Casual T-shirt', 'Jeans', 'Sneakers'],
        purchase_frequency='high',
        avg_price=0.035,
        recency=3
    )
    
    # 페르소나 생성
    persona = create_persona(test_metadata)
    
    print("=" * 60)
    print("Generated Persona:")
    print("=" * 60)
    print(persona)
    print("=" * 60)


if __name__ == "__main__":
    main()
