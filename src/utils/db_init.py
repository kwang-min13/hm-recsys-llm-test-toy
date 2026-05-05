"""
DuckDB 초기화 및 유틸리티 모듈

이 모듈은 DuckDB 연결 및 대용량 데이터 처리를 위한 설정을 제공합니다.
"""

import duckdb
from pathlib import Path
from typing import Optional


class DuckDBManager:
    """DuckDB 연결 및 설정 관리 클래스"""
    
    def __init__(self, db_path: str = 'local_helix.db'):
        """
        DuckDB 매니저 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.con: Optional[duckdb.DuckDBPyConnection] = None
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        DuckDB 연결 생성 및 최적화 설정
        
        Returns:
            DuckDB 연결 객체
        """
        if self.con is None:
            self.con = duckdb.connect(self.db_path)
            self._configure_settings()
        
        return self.con
    
    def _configure_settings(self):
        """대용량 데이터 처리를 위한 DuckDB 설정"""
        if self.con is None:
            raise RuntimeError("연결이 초기화되지 않았습니다.")
        
        # 메모리 제한 설정 (8GB)
        self.con.execute("SET memory_limit='8GB'")
        
        # 스레드 수 설정 (CPU 코어 수에 따라 자동 조정)
        self.con.execute("SET threads TO 4")
        
        # 진행률 표시 활성화
        self.con.execute("SET enable_progress_bar=true")
        
        # Parquet 파일 읽기 최적화
        self.con.execute("SET preserve_insertion_order=false")
    
    def close(self):
        """연결 종료"""
        if self.con is not None:
            self.con.close()
            self.con = None
    
    def execute(self, query: str):
        """
        SQL 쿼리 실행
        
        Args:
            query: 실행할 SQL 쿼리
            
        Returns:
            쿼리 실행 결과
        """
        if self.con is None:
            self.connect()
        
        return self.con.execute(query)
    
    def __enter__(self):
        """Context manager 진입"""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.close()


def create_database_schema(con: duckdb.DuckDBPyConnection):
    """
    초기 데이터베이스 스키마 생성
    
    Args:
        con: DuckDB 연결 객체
    """
    # User Features 테이블
    con.execute("""
        CREATE TABLE IF NOT EXISTS user_features (
            customer_id VARCHAR PRIMARY KEY,
            avg_purchase_hour DOUBLE,
            preferred_category VARCHAR,
            recency INTEGER,
            purchase_count INTEGER,
            purchase_frequency VARCHAR
        )
    """)
    
    # Item Features 테이블
    con.execute("""
        CREATE TABLE IF NOT EXISTS item_features (
            article_id VARCHAR PRIMARY KEY,
            popularity_rank INTEGER,
            peak_hour INTEGER,
            sales_count INTEGER,
            category VARCHAR
        )
    """)
    
    print("✓ 데이터베이스 스키마 생성 완료")


def test_connection():
    """DuckDB 연결 테스트"""
    try:
        manager = DuckDBManager(':memory:')
        con = manager.connect()
        
        # 간단한 쿼리 테스트
        result = con.execute("SELECT 'Connection OK' as status").fetchone()
        print(f"✓ DuckDB 연결 테스트 성공: {result[0]}")
        
        manager.close()
        return True
    except Exception as e:
        print(f"✗ DuckDB 연결 테스트 실패: {str(e)}")
        return False


if __name__ == "__main__":
    # 테스트 실행
    print("DuckDB 초기화 테스트")
    print("=" * 40)
    
    if test_connection():
        print("\n데이터베이스 스키마 생성 테스트")
        manager = DuckDBManager(':memory:')
        con = manager.connect()
        create_database_schema(con)
        manager.close()
        print("\n✓ 모든 테스트 통과")
    else:
        print("\n✗ 테스트 실패")
