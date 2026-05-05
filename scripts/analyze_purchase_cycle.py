"""
유저별 평균 구매 주기 분석 스크립트

이 스크립트는 DuckDB를 사용하여 유저들의 평균 구매 주기를 계산합니다.
구매 주기는 연속된 구매 사이의 평균 일수로 정의됩니다.
"""

import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_purchase_cycle(db_path: str = 'local_helix.db'):
    """
    유저별 평균 구매 주기 분석
    
    Args:
        db_path: DuckDB 데이터베이스 경로
    """
    logger.info("=" * 80)
    logger.info("유저 평균 구매 주기 분석 시작")
    logger.info("=" * 80)
    
    con = duckdb.connect(db_path)
    
    try:
        # 메모리 설정
        con.execute("SET memory_limit='8GB'")
        con.execute("SET threads TO 4")
        
        # 1. 전체 유저의 평균 구매 주기 계산
        logger.info("\n[1] 전체 유저 평균 구매 주기 계산 중...")
        
        query_overall = """
        WITH user_purchases AS (
            SELECT 
                customer_id,
                t_dat::DATE as purchase_date,
                LAG(t_dat::DATE) OVER (PARTITION BY customer_id ORDER BY t_dat) as prev_purchase_date
            FROM read_csv_auto('data/transactions_train.csv')
            WHERE customer_id IS NOT NULL
        ),
        purchase_intervals AS (
            SELECT 
                customer_id,
                DATE_DIFF('day', prev_purchase_date, purchase_date) as days_between_purchases
            FROM user_purchases
            WHERE prev_purchase_date IS NOT NULL
        ),
        user_avg_cycles AS (
            SELECT 
                customer_id,
                AVG(days_between_purchases) as avg_purchase_cycle,
                COUNT(*) as num_intervals,
                MIN(days_between_purchases) as min_cycle,
                MAX(days_between_purchases) as max_cycle
            FROM purchase_intervals
            GROUP BY customer_id
            HAVING COUNT(*) >= 1  -- 최소 2번 이상 구매한 유저
        )
        SELECT 
            COUNT(*) as total_users_with_repeat_purchases,
            ROUND(AVG(avg_purchase_cycle), 2) as overall_avg_purchase_cycle,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_purchase_cycle), 2) as median_purchase_cycle,
            ROUND(MIN(avg_purchase_cycle), 2) as min_avg_cycle,
            ROUND(MAX(avg_purchase_cycle), 2) as max_avg_cycle,
            ROUND(STDDEV(avg_purchase_cycle), 2) as stddev_purchase_cycle
        FROM user_avg_cycles
        """
        
        result = con.execute(query_overall).fetchone()
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 전체 유저 평균 구매 주기 통계")
        logger.info("=" * 80)
        logger.info(f"재구매 유저 수: {result[0]:,}명")
        logger.info(f"평균 구매 주기: {result[1]:.2f}일")
        logger.info(f"중앙값 구매 주기: {result[2]:.2f}일")
        logger.info(f"최소 평균 주기: {result[3]:.2f}일")
        logger.info(f"최대 평균 주기: {result[4]:.2f}일")
        logger.info(f"표준편차: {result[5]:.2f}일")
        logger.info("=" * 80)
        
        # 2. 구매 주기 분포 분석
        logger.info("\n[2] 구매 주기 분포 분석 중...")
        
        query_distribution = """
        WITH user_purchases AS (
            SELECT 
                customer_id,
                t_dat::DATE as purchase_date,
                LAG(t_dat::DATE) OVER (PARTITION BY customer_id ORDER BY t_dat) as prev_purchase_date
            FROM read_csv_auto('data/transactions_train.csv')
            WHERE customer_id IS NOT NULL
        ),
        purchase_intervals AS (
            SELECT 
                customer_id,
                DATE_DIFF('day', prev_purchase_date, purchase_date) as days_between_purchases
            FROM user_purchases
            WHERE prev_purchase_date IS NOT NULL
        ),
        user_avg_cycles AS (
            SELECT 
                customer_id,
                AVG(days_between_purchases) as avg_purchase_cycle
            FROM purchase_intervals
            GROUP BY customer_id
        )
        SELECT 
            CASE 
                WHEN avg_purchase_cycle <= 7 THEN '1주 이내'
                WHEN avg_purchase_cycle <= 14 THEN '1-2주'
                WHEN avg_purchase_cycle <= 30 THEN '2주-1개월'
                WHEN avg_purchase_cycle <= 60 THEN '1-2개월'
                WHEN avg_purchase_cycle <= 90 THEN '2-3개월'
                ELSE '3개월 이상'
            END as cycle_range,
            COUNT(*) as user_count,
            ROUND(AVG(avg_purchase_cycle), 2) as avg_cycle_in_range
        FROM user_avg_cycles
        GROUP BY cycle_range
        ORDER BY 
            CASE cycle_range
                WHEN '1주 이내' THEN 1
                WHEN '1-2주' THEN 2
                WHEN '2주-1개월' THEN 3
                WHEN '1-2개월' THEN 4
                WHEN '2-3개월' THEN 5
                ELSE 6
            END
        """
        
        distribution = con.execute(query_distribution).fetchall()
        
        logger.info("\n" + "=" * 80)
        logger.info("📈 구매 주기 분포")
        logger.info("=" * 80)
        total_users = sum(row[1] for row in distribution)
        for row in distribution:
            cycle_range, user_count, avg_cycle = row
            percentage = (user_count / total_users) * 100
            logger.info(f"{cycle_range:15s}: {user_count:8,}명 ({percentage:5.2f}%) - 평균 {avg_cycle:6.2f}일")
        logger.info("=" * 80)
        
        # 3. 구매 빈도별 평균 구매 주기
        logger.info("\n[3] 구매 빈도별 평균 구매 주기 분석 중...")
        
        query_by_frequency = """
        WITH user_purchases AS (
            SELECT 
                customer_id,
                t_dat::DATE as purchase_date,
                LAG(t_dat::DATE) OVER (PARTITION BY customer_id ORDER BY t_dat) as prev_purchase_date
            FROM read_csv_auto('data/transactions_train.csv')
            WHERE customer_id IS NOT NULL
        ),
        purchase_intervals AS (
            SELECT 
                customer_id,
                DATE_DIFF('day', prev_purchase_date, purchase_date) as days_between_purchases
            FROM user_purchases
            WHERE prev_purchase_date IS NOT NULL
        ),
        user_stats AS (
            SELECT 
                customer_id,
                AVG(days_between_purchases) as avg_purchase_cycle,
                COUNT(*) + 1 as total_purchases  -- +1 because intervals = purchases - 1
            FROM purchase_intervals
            GROUP BY customer_id
        )
        SELECT 
            CASE 
                WHEN total_purchases >= 20 THEN 'VIP (20회 이상)'
                WHEN total_purchases >= 10 THEN '고빈도 (10-19회)'
                WHEN total_purchases >= 5 THEN '중빈도 (5-9회)'
                ELSE '저빈도 (2-4회)'
            END as frequency_segment,
            COUNT(*) as user_count,
            ROUND(AVG(avg_purchase_cycle), 2) as avg_cycle,
            ROUND(MIN(avg_purchase_cycle), 2) as min_cycle,
            ROUND(MAX(avg_purchase_cycle), 2) as max_cycle
        FROM user_stats
        GROUP BY frequency_segment
        ORDER BY 
            CASE frequency_segment
                WHEN 'VIP (20회 이상)' THEN 1
                WHEN '고빈도 (10-19회)' THEN 2
                WHEN '중빈도 (5-9회)' THEN 3
                ELSE 4
            END
        """
        
        frequency_results = con.execute(query_by_frequency).fetchall()
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 구매 빈도별 평균 구매 주기")
        logger.info("=" * 80)
        logger.info(f"{'세그먼트':20s} {'유저 수':>12s} {'평균 주기':>12s} {'최소':>10s} {'최대':>10s}")
        logger.info("-" * 80)
        for row in frequency_results:
            segment, count, avg, min_c, max_c = row
            logger.info(f"{segment:20s} {count:12,}명 {avg:11.2f}일 {min_c:9.2f}일 {max_c:9.2f}일")
        logger.info("=" * 80)
        
        # 4. 최근 활동 유저의 구매 주기
        logger.info("\n[4] 최근 활동 유저 (최근 30일 이내 구매) 분석 중...")
        
        query_recent = """
        WITH recent_users AS (
            SELECT DISTINCT customer_id
            FROM read_csv_auto('data/transactions_train.csv')
            WHERE t_dat >= (SELECT MAX(t_dat) - INTERVAL '30 days' FROM read_csv_auto('data/transactions_train.csv'))
        ),
        user_purchases AS (
            SELECT 
                t.customer_id,
                t.t_dat::DATE as purchase_date,
                LAG(t.t_dat::DATE) OVER (PARTITION BY t.customer_id ORDER BY t.t_dat) as prev_purchase_date
            FROM read_csv_auto('data/transactions_train.csv') t
            INNER JOIN recent_users ru ON t.customer_id = ru.customer_id
        ),
        purchase_intervals AS (
            SELECT 
                customer_id,
                DATE_DIFF('day', prev_purchase_date, purchase_date) as days_between_purchases
            FROM user_purchases
            WHERE prev_purchase_date IS NOT NULL
        )
        SELECT 
            COUNT(DISTINCT customer_id) as active_users,
            ROUND(AVG(days_between_purchases), 2) as avg_cycle,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_between_purchases), 2) as median_cycle
        FROM purchase_intervals
        """
        
        recent_result = con.execute(query_recent).fetchone()
        
        logger.info("\n" + "=" * 80)
        logger.info("🔥 최근 활동 유저 (최근 30일 이내)")
        logger.info("=" * 80)
        logger.info(f"활성 유저 수: {recent_result[0]:,}명")
        logger.info(f"평균 구매 주기: {recent_result[1]:.2f}일")
        logger.info(f"중앙값 구매 주기: {recent_result[2]:.2f}일")
        logger.info("=" * 80)
        
        logger.info("\n✅ 분석 완료!")
        
    except Exception as e:
        logger.error(f"❌ 에러 발생: {str(e)}")
        raise
    finally:
        con.close()


if __name__ == "__main__":
    analyze_purchase_cycle()
