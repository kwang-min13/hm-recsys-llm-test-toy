"""
User Feature Generation Module

이 모듈은 H&M 트랜잭션 데이터로부터 유저별 Feature를 생성합니다.
DuckDB를 활용하여 대용량 데이터를 효율적으로 처리합니다.
"""

import duckdb
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserFeatureGenerator:
    """유저 Feature 생성 클래스"""
    
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
            # 메모리 설정
            self.con.execute("SET memory_limit='8GB'")
            self.con.execute("SET threads TO 4")
        return self.con
    
    def create_user_features(self, 
                            transactions_path: str = 'data/transactions_train.csv',
                            output_path: str = 'data/features/user_features.parquet',
                            lookback_days: int = 28):
        """
        유저별 Feature 생성
        
        Args:
            transactions_path: 트랜잭션 CSV 파일 경로
            output_path: 출력 Parquet 파일 경로
            lookback_days: Feature 계산 기간 (일)
        """
        logger.info("유저 Feature 생성 시작...")
        
        con = self.connect()
        
        # 트랜잭션 데이터 로드 및 Feature 생성
        query = f"""
        CREATE OR REPLACE TABLE user_features AS
        WITH recent_transactions AS (
            SELECT 
                customer_id,
                t_dat,
                article_id,
                price
            FROM read_csv_auto('{transactions_path}')
            WHERE t_dat >= (SELECT MAX(t_dat) - INTERVAL '{lookback_days} days' FROM read_csv_auto('{transactions_path}'))
        ),
        user_stats AS (
            SELECT 
                customer_id,
                COUNT(*) as purchase_count,
                COUNT(DISTINCT article_id) as unique_items,
                AVG(price) as avg_price,
                MAX(t_dat) as last_purchase_date,
                MIN(t_dat) as first_purchase_date
            FROM recent_transactions
            GROUP BY customer_id
        ),
        max_date AS (
            SELECT MAX(t_dat) as max_date FROM recent_transactions
        )
        SELECT 
            us.customer_id,
            us.purchase_count,
            us.unique_items,
            ROUND(us.avg_price, 2) as avg_price,
            DATE_DIFF('day', us.last_purchase_date, md.max_date) as recency,
            CASE 
                WHEN us.purchase_count >= 10 THEN 'high'
                WHEN us.purchase_count >= 5 THEN 'medium'
                ELSE 'low'
            END as purchase_frequency,
            us.last_purchase_date,
            us.first_purchase_date
        FROM user_stats us
        CROSS JOIN max_date md
        """
        
        logger.info("SQL 쿼리 실행 중...")
        con.execute(query)
        
        # Parquet로 저장
        logger.info(f"Feature를 {output_path}에 저장 중...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        con.execute(f"""
            COPY user_features TO '{output_path}' (FORMAT PARQUET)
        """)
        
        # 통계 출력
        stats = con.execute("""
            SELECT 
                COUNT(*) as total_users,
                AVG(purchase_count) as avg_purchases,
                COUNT(CASE WHEN purchase_frequency = 'high' THEN 1 END) as high_freq_users,
                COUNT(CASE WHEN purchase_frequency = 'medium' THEN 1 END) as medium_freq_users,
                COUNT(CASE WHEN purchase_frequency = 'low' THEN 1 END) as low_freq_users
            FROM user_features
        """).fetchone()
        
        logger.info("=" * 60)
        logger.info("유저 Feature 생성 완료!")
        logger.info(f"총 유저 수: {stats[0]:,}")
        logger.info(f"평균 구매 횟수: {stats[1]:.2f}")
        logger.info(f"구매 빈도 분포:")
        logger.info(f"  - High: {stats[2]:,} ({stats[2]/stats[0]*100:.1f}%)")
        logger.info(f"  - Medium: {stats[3]:,} ({stats[3]/stats[0]*100:.1f}%)")
        logger.info(f"  - Low: {stats[4]:,} ({stats[4]/stats[0]*100:.1f}%)")
        logger.info("=" * 60)
        
        return output_path
    
    def close(self):
        """연결 종료"""
        if self.con is not None:
            self.con.close()
            self.con = None


def main():
    """메인 실행 함수"""
    generator = UserFeatureGenerator()
    
    try:
        output_path = generator.create_user_features()
        logger.info(f"✓ 유저 Feature가 {output_path}에 저장되었습니다.")
    except Exception as e:
        logger.error(f"✗ 에러 발생: {str(e)}")
        raise
    finally:
        generator.close()


if __name__ == "__main__":
    main()
