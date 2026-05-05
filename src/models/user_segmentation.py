"""
User Segmentation Module

Classifies users into segments based on activity level:
- ACTIVE: Users with sufficient purchase history (3+ purchases in last 60 days)
- LOW_ACTIVITY: New or inactive users with minimal purchase history

This enables adaptive recommendation strategies:
- Active users → CF + Ranker (personalized)
- Low-activity users → Popularity + Recency (fallback)
"""

from __future__ import annotations

import duckdb
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserSegment(Enum):
    """User segment classification"""
    ACTIVE = "active"
    LOW_ACTIVITY = "low_activity"


@dataclass
class UserActivityMetrics:
    """User activity metrics for segmentation"""
    user_id: str
    purchase_count: int
    purchase_count_60d: int  # Purchases in last 60 days
    recency: int  # Days since last purchase
    segment: UserSegment


class UserSegmenter:
    """
    Classifies users into segments based on activity level.
    
    Segmentation criteria:
    - ACTIVE: purchase_count_60d >= threshold (default: 3)
    - LOW_ACTIVITY: purchase_count_60d < threshold
    """
    
    def __init__(
        self,
        db_path: str = "local_helix.db",
        transactions_path: str = "data/transactions_train.csv",
        user_features_path: str = "data/features/user_features.parquet",
        activity_threshold: int = 3,
        activity_window_days: int = 60,
        memory_limit: str = "8GB",
        threads: int = 4,
    ):
        self.db_path = db_path
        self.transactions_path = transactions_path
        self.user_features_path = user_features_path
        self.activity_threshold = int(activity_threshold)
        self.activity_window_days = int(activity_window_days)
        self.memory_limit = memory_limit
        self.threads = threads
        
        self.con: Optional[duckdb.DuckDBPyConnection] = None
        self._cache_ready = False
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        """Connect to DuckDB and prepare cache"""
        if self.con is None:
            # Use in-memory database for better performance
            self.con = duckdb.connect(":memory:")
            self.con.execute(f"SET memory_limit='{self.memory_limit}'")
            self.con.execute(f"SET threads TO {int(self.threads)}")
        
        if not self._cache_ready:
            self._prepare_cache()
        
        return self.con
    
    def _prepare_cache(self) -> None:
        """Prepare views for user activity calculation"""
        con = self.con
        assert con is not None
        
        # User features view
        con.execute("DROP VIEW IF EXISTS v_user_features")
        con.execute(
            f"""
            CREATE VIEW v_user_features AS
            SELECT
                customer_id::VARCHAR AS customer_id,
                purchase_count,
                recency
            FROM read_parquet('{self.user_features_path}')
            """
        )
        
        # Transactions view
        con.execute("DROP VIEW IF EXISTS v_transactions")
        con.execute(
            f"""
            CREATE VIEW v_transactions AS
            SELECT
                customer_id::VARCHAR AS customer_id,
                article_id::VARCHAR AS article_id,
                CAST(t_dat AS DATE) AS t_dat
            FROM read_csv_auto('{self.transactions_path}', header=true)
            """
        )
        
        # Max date for window calculation
        con.execute("DROP VIEW IF EXISTS v_max_date")
        con.execute(
            """
            CREATE VIEW v_max_date AS
            SELECT MAX(t_dat) AS max_date
            FROM v_transactions
            """
        )
        
        self._cache_ready = True
        logger.info("User segmentation cache ready")
    
    def get_user_activity_metrics(self, user_id: str) -> Optional[UserActivityMetrics]:
        """
        Get detailed activity metrics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            UserActivityMetrics or None if user not found
        """
        con = self.connect()
        
        query = f"""
        WITH
        maxd AS (SELECT max_date FROM v_max_date),
        user_stats AS (
            SELECT
                uf.customer_id,
                uf.purchase_count,
                uf.recency,

                -- ✅ 구매 '이벤트 수' (행 수)로 계산
                COALESCE(COUNT(t.article_id), 0) AS purchase_count_60d

            FROM v_user_features uf
            LEFT JOIN v_transactions t
                ON t.customer_id = uf.customer_id
               AND t.t_dat >= (SELECT max_date - INTERVAL '{self.activity_window_days} days' FROM maxd)
            WHERE uf.customer_id = ?
            GROUP BY uf.customer_id, uf.purchase_count, uf.recency
        )
        SELECT
            customer_id,
            purchase_count,
            COALESCE(purchase_count_60d, 0) AS purchase_count_60d,
            recency
        FROM user_stats
        """
        
        result = con.execute(query, [str(user_id)]).fetchone()
        
        if result is None:
            logger.warning(f"User {user_id} not found in user features")
            return None
        
        customer_id, purchase_count, purchase_count_60d, recency = result
        
        # Determine segment
        segment = (
            UserSegment.ACTIVE
            if purchase_count_60d >= self.activity_threshold
            else UserSegment.LOW_ACTIVITY
        )
        
        return UserActivityMetrics(
            user_id=str(customer_id),
            purchase_count=int(purchase_count) if purchase_count else 0,
            purchase_count_60d=int(purchase_count_60d) if purchase_count_60d else 0,
            recency=int(recency) if recency else 999,
            segment=segment,
        )
    
    def classify_user(self, user_id: str) -> UserSegment:
        """
        Classify user into segment (ACTIVE or LOW_ACTIVITY).
        
        Args:
            user_id: User ID
            
        Returns:
            UserSegment (defaults to LOW_ACTIVITY if user not found)
        """
        metrics = self.get_user_activity_metrics(user_id)
        
        if metrics is None:
            logger.warning(f"User {user_id} not found, defaulting to LOW_ACTIVITY segment")
            return UserSegment.LOW_ACTIVITY
        
        logger.info(
            f"User {user_id}: {metrics.purchase_count_60d} purchases in {self.activity_window_days}d "
            f"→ {metrics.segment.value}"
        )
        
        return metrics.segment
    
    def close(self) -> None:
        """Close database connection"""
        if self.con is not None:
            self.con.close()
            self.con = None
        self._cache_ready = False


def main():
    """Test user segmentation"""
    import duckdb
    
    segmenter = UserSegmenter(
        activity_threshold=3,
        activity_window_days=60,
    )
    
    try:
        # Get a sample user
        con = duckdb.connect(":memory:")
        sample_user = con.execute(
            """
            SELECT customer_id
            FROM read_parquet('data/features/user_features.parquet')
            LIMIT 1
            """
        ).fetchone()[0]
        con.close()
        
        logger.info("=" * 60)
        logger.info(f"Testing user segmentation for: {sample_user}")
        logger.info("=" * 60)
        
        # Get metrics
        metrics = segmenter.get_user_activity_metrics(sample_user)
        
        if metrics:
            logger.info(f"User ID: {metrics.user_id}")
            logger.info(f"Total purchases: {metrics.purchase_count}")
            logger.info(f"Purchases (60d): {metrics.purchase_count_60d}")
            logger.info(f"Recency: {metrics.recency} days")
            logger.info(f"Segment: {metrics.segment.value}")
        
        # Classify
        segment = segmenter.classify_user(sample_user)
        logger.info(f"\nClassification: {segment.value}")
        
    finally:
        segmenter.close()


if __name__ == "__main__":
    main()
