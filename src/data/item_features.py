"""
Item Feature Generation Module

이 모듈은 H&M 트랜잭션 데이터로부터 상품별 Feature를 생성합니다.
"""

import duckdb
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ItemFeatureGenerator:
    """상품 Feature 생성 클래스"""
    
    def __init__(self, db_path: str = 'local_helix.db'):
        """
        초기화
        
        Args:
            db_path: DuckDB 데이터베이스 경로
        """
        self.db_path = db_path
        self.con: Optional[duckdb.DuckDBPyConnection] = None
    
    def connect(self):
        """DuckDB 연결"""
        if self.con is None:
            self.con = duckdb.connect(self.db_path)
            self.con.execute("SET memory_limit='8GB'")
            self.con.execute("SET threads TO 4")
        return self.con
    
    def create_item_features(self,
                            transactions_path: str = 'data/transactions_train.csv',
                            articles_path: str = 'data/articles.csv',
                            output_path: str = 'data/features/item_features.parquet',
                            lookback_days: int = 7):
        """
        상품별 Feature 생성
        
        Args:
            transactions_path: 트랜잭션 CSV 파일 경로
            articles_path: 상품 정보 CSV 파일 경로
            output_path: 출력 Parquet 파일 경로
            lookback_days: Feature 계산 기간 (일)
        """
        logger.info("상품 Feature 생성 시작...")
        
        con = self.connect()
        
        # 상품 Feature 생성
        query = f"""
        CREATE OR REPLACE TABLE item_features AS
        WITH recent_transactions AS (
            SELECT 
                article_id,
                t_dat,
                price,
                customer_id
            FROM read_csv_auto('{transactions_path}')
            WHERE t_dat >= (SELECT MAX(t_dat) - INTERVAL '{lookback_days} days' FROM read_csv_auto('{transactions_path}'))
        ),
        item_stats AS (
            SELECT 
                article_id,
                COUNT(*) as sales_count,
                COUNT(DISTINCT customer_id) as unique_customers,
                AVG(price) as avg_price,
                MAX(t_dat) as last_sold_date
            FROM recent_transactions
            GROUP BY article_id
        ),
        item_popularity AS (
            SELECT 
                a.article_id,
                a.prod_name,
                sales_count,
                ROW_NUMBER() OVER (ORDER BY sales_count DESC) as popularity_rank
            FROM item_stats ist
            INNER JOIN read_csv_auto('{articles_path}') a ON ist.article_id = a.article_id
        )
        SELECT 
            ist.article_id,
            ip.prod_name,
            ip.popularity_rank,
            ist.sales_count,
            ist.unique_customers,
            ROUND(ist.avg_price, 2) as avg_price,
            ist.last_sold_date
        FROM item_stats ist
        JOIN item_popularity ip ON ist.article_id = ip.article_id
        ORDER BY ip.popularity_rank
        """
        
        logger.info("SQL 쿼리 실행 중...")
        con.execute(query)
        
        # Parquet로 저장
        logger.info(f"Feature를 {output_path}에 저장 중...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        con.execute(f"""
            COPY item_features TO '{output_path}' (FORMAT PARQUET)
        """)
        
        # 통계 출력
        stats = con.execute("""
            SELECT 
                COUNT(*) as total_items,
                AVG(sales_count) as avg_sales,
                MAX(sales_count) as max_sales,
                MIN(sales_count) as min_sales,
                COUNT(CASE WHEN popularity_rank <= 100 THEN 1 END) as top_100_items
            FROM item_features
        """).fetchone()
        
        logger.info("=" * 60)
        logger.info("상품 Feature 생성 완료!")
        logger.info(f"총 상품 수: {stats[0]:,}")
        logger.info(f"평균 판매량: {stats[1]:.2f}")
        logger.info(f"최대 판매량: {stats[2]:,}")
        logger.info(f"최소 판매량: {stats[3]:,}")
        logger.info(f"Top 100 상품 수: {stats[4]:,}")
        logger.info("=" * 60)
        
        return output_path
    
    def close(self):
        """연결 종료"""
        if self.con is not None:
            self.con.close()
            self.con = None


def main():
    """메인 실행 함수"""
    generator = ItemFeatureGenerator()
    
    try:
        output_path = generator.create_item_features()
        logger.info(f"✓ 상품 Feature가 {output_path}에 저장되었습니다.")
    except Exception as e:
        logger.error(f"✗ 에러 발생: {str(e)}")
        raise
    finally:
        generator.close()


if __name__ == "__main__":
    main()
