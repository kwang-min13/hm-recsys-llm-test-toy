"""
Feature Store Module

Feature 관리 및 조회를 위한 중앙화된 저장소
"""

import duckdb
import polars as pl
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureStore:
    """Feature Store 클래스"""
    
    def __init__(self, 
                 db_path: str = 'local_helix.db',
                 features_dir: str = 'data/features'):
        """
        초기화
        
        Args:
            db_path: DuckDB 데이터베이스 경로
            features_dir: Feature 파일 저장 디렉토리
        """
        self.db_path = db_path
        self.features_dir = Path(features_dir)
        self.con: Optional[duckdb.DuckDBPyConnection] = None
    
    def connect(self):
        """DuckDB 연결 (메모리 데이터베이스 사용)"""
        if self.con is None:
            # Use in-memory database to avoid file locking
            self.con = duckdb.connect(":memory:")
            # Attach the file database in read-only mode if it exists
            if self.db_path != ":memory:":
                try:
                    self.con.execute(f"ATTACH '{self.db_path}' AS filedb (READ_ONLY)")
                except Exception:
                    # If attach fails, continue with in-memory only
                    pass
        return self.con
    
    def get_user_features(self, user_ids: Optional[List[str]] = None) -> pl.DataFrame:
        """
        유저 Feature 조회
        
        Args:
            user_ids: 조회할 유저 ID 리스트 (None이면 전체)
            
        Returns:
            Polars DataFrame
        """
        con = self.connect()
        user_features_path = self.features_dir / 'user_features.parquet'
        
        if not user_features_path.exists():
            raise FileNotFoundError(f"User features not found: {user_features_path}")
        
        if user_ids is None:
            # 전체 유저 Feature 조회
            query = f"SELECT * FROM read_parquet('{user_features_path}')"
            result = con.execute(query).fetch_df()
        else:
            # 특정 유저 Feature 조회 (SQL injection 방지)
            query = f"""
                SELECT * FROM read_parquet('{user_features_path}')
                WHERE customer_id IN (SELECT unnest(?))
            """
            result = con.execute(query, [user_ids]).fetch_df()
            if "purchase_frequency_encoded" not in result.columns:
                mapping = {"low": 0, "medium": 1, "high": 2}
                result["purchase_frequency_encoded"] = (
                    result["purchase_frequency"].map(mapping).fillna(0).astype("int64")
                )
            
        return pl.from_pandas(result)
    
    def get_item_features(self, item_ids: Optional[List[str]] = None) -> pl.DataFrame:
        """
        상품 Feature 조회
        
        Args:
            item_ids: 조회할 상품 ID 리스트 (None이면 전체)
            
        Returns:
            Polars DataFrame
        """
        con = self.connect()
        item_features_path = self.features_dir / 'item_features.parquet'
        
        if not item_features_path.exists():
            raise FileNotFoundError(f"Item features not found: {item_features_path}")
        
        if item_ids is None:
            # 전체 상품 Feature 조회
            query = f"SELECT * FROM read_parquet('{item_features_path}')"
            result = con.execute(query).fetch_df()
        else:
            # 특정 상품 Feature 조회 (SQL injection 방지)
            query = f"""
                SELECT * FROM read_parquet('{item_features_path}')
                WHERE article_id IN (SELECT unnest(?))
            """
            result = con.execute(query, [item_ids]).fetch_df()
        
        return pl.from_pandas(result)
    
    def get_top_items(self, top_k: int = 100) -> pl.DataFrame:
        """
        인기 상품 Top K 조회
        
        Args:
            top_k: 조회할 상품 수
            
        Returns:
            Polars DataFrame
        """
        con = self.connect()
        item_features_path = self.features_dir / 'item_features.parquet'
        
        query = f"""
            SELECT * FROM read_parquet('{item_features_path}')
            WHERE popularity_rank <= {top_k}
            ORDER BY popularity_rank
        """
        
        result = con.execute(query).fetch_df()
        return pl.from_pandas(result)
    
    def refresh_features(self):
        """
        모든 Feature 재생성
        """
        from .user_features import UserFeatureGenerator
        from .item_features import ItemFeatureGenerator
        
        logger.info("Feature 재생성 시작...")
        
        # User Features 생성
        user_gen = UserFeatureGenerator(self.db_path)
        try:
            user_gen.create_user_features(
                output_path=str(self.features_dir / 'user_features.parquet')
            )
        finally:
            user_gen.close()
        
        # Item Features 생성
        item_gen = ItemFeatureGenerator(self.db_path)
        try:
            item_gen.create_item_features(
                output_path=str(self.features_dir / 'item_features.parquet')
            )
        finally:
            item_gen.close()
        
        logger.info("✓ 모든 Feature 재생성 완료!")
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """
        Feature 통계 정보 조회
        
        Returns:
            통계 정보 딕셔너리
        """
        con = self.connect()
        
        stats = {}
        
        # User Features 통계
        user_features_path = self.features_dir / 'user_features.parquet'
        if user_features_path.exists():
            user_stats = con.execute(f"""
                SELECT 
                    COUNT(*) as total_users,
                    AVG(purchase_count) as avg_purchases,
                FROM read_parquet('{user_features_path}')
            """).fetchone()
            
            stats['users'] = {
                'total': user_stats[0],
                'avg_purchases': round(user_stats[1], 2),
            }
        
        # Item Features 통계
        item_features_path = self.features_dir / 'item_features.parquet'
        if item_features_path.exists():
            item_stats = con.execute(f"""
                SELECT 
                    COUNT(*) as total_items,
                    AVG(sales_count) as avg_sales,
                    MAX(sales_count) as max_sales
                FROM read_parquet('{item_features_path}')
            """).fetchone()
            
            stats['items'] = {
                'total': item_stats[0],
                'avg_sales': round(item_stats[1], 2),
                'max_sales': item_stats[2]
            }
        
        return stats
    
    def close(self):
        """연결 종료"""
        if self.con is not None:
            self.con.close()
            self.con = None


def main():
    """테스트 실행"""
    store = FeatureStore()
    
    try:
        # Feature 통계 출력
        stats = store.get_feature_stats()
        
        logger.info("=" * 60)
        logger.info("Feature Store 통계")
        logger.info("=" * 60)
        
        if 'users' in stats:
            logger.info(f"총 유저 수: {stats['users']['total']:,}")
            logger.info(f"평균 구매 횟수: {stats['users']['avg_purchases']}")
        
        if 'items' in stats:
            logger.info(f"총 상품 수: {stats['items']['total']:,}")
            logger.info(f"평균 판매량: {stats['items']['avg_sales']}")
            logger.info(f"최대 판매량: {stats['items']['max_sales']:,}")
        
        logger.info("=" * 60)
        
    finally:
        store.close()


if __name__ == "__main__":
    main()
