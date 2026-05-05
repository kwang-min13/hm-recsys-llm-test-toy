"""
Dataset Creation Module (Improved)

목표:
- 2-Stage 추천에서 Ranking 학습에 맞는 데이터 생성
- 유저별 query(group) 구조 제공: user 당 (1 positive + N negatives)
- 검증셋에서 positive-only 발생 방지

출력:
- features: pl.DataFrame
- labels:   pl.DataFrame (label: 0/1)
- group:    List[int]  (각 유저별 row 수)
"""

from __future__ import annotations

import duckdb
import polars as pl
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_ranking_dataset(
    db_path: str = ":memory:",
    sample_users: int = 10000,
    pos_per_user: int = 3,
    negative_per_user: int = 50,
    window_days: int = 28,
    popularity_pool: int = 2000,
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, List[int]]:

    """
    Args:
        db_path: DuckDB 경로
        sample_users: 샘플링할 유저 수
        negative_per_user: 유저당 negative 개수
        window_days: 최근 거래 window
        popularity_pool: negative 후보 인기 pool 크기 (너무 작으면 충돌↑)
        seed: 재현성 seed

    Returns:
        (features, labels, group)
    """
    logger.info("Ranking dataset 생성 시작...")
    con = duckdb.connect(db_path)
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads TO 4")

    # DuckDB random seed 고정(가능한 경우)
    # DuckDB 버전에 따라 안 먹을 수 있으나, ORDER BY random()가 조금이라도 안정됨
    try:
        con.execute(f"SET seed={int(seed)}")
    except Exception:
        pass
    
    # ✅ pos 늘어났을 때 난이도 유지용
    pos_per_user = int(pos_per_user)
    negative_per_user = int(negative_per_user)
    effective_negative = negative_per_user * pos_per_user

    query = f"""
    WITH
    -- 1) 최근 window로 transaction 제한 (1회 스캔 + window 통일)
    t_all AS (
        SELECT
            customer_id::VARCHAR AS customer_id,
            article_id::VARCHAR  AS article_id,
            CAST(t_dat AS DATE)  AS t_dat
        FROM read_csv_auto('data/transactions_train.csv', header=true)
    ),
    t_recent AS (
        SELECT *
        FROM t_all
        WHERE t_dat >= (SELECT MAX(t_dat) - INTERVAL '{int(window_days)} days' FROM t_all)
    ),

    -- 2) 유저 샘플링: 최근 window에서 활동한 유저
    sampled_users AS (
        SELECT DISTINCT customer_id
        FROM t_recent
        ORDER BY random()
        LIMIT {int(sample_users)}
    ),

    -- 3) 유저별 positive: 가장 최근 구매 {pos_per_user}개(타이브레이크 포함)
     user_pos_raw AS (
        SELECT customer_id, article_id, 1 AS label
        FROM (
            SELECT
                tr.customer_id,
                tr.article_id,
                tr.t_dat,
                ROW_NUMBER() OVER (
                    PARTITION BY tr.customer_id
                    ORDER BY tr.t_dat DESC, tr.article_id ASC
                ) AS rn
            FROM t_recent tr
            INNER JOIN sampled_users su
                ON tr.customer_id = su.customer_id
        )
        WHERE rn <= {int(pos_per_user)}
    ),
    
    pos_users AS (
        SELECT customer_id
        FROM user_pos_raw
        GROUP BY customer_id
        HAVING COUNT(*) == {pos_per_user}
    ),

    user_pos AS (
        SELECT upr.*
        FROM user_pos_raw upr
        INNER JOIN pos_users pu USING(customer_id)
    ),

    -- 4) 유저 구매 이력(최근 window): negative에서 제외
    user_purchased AS (
        SELECT DISTINCT customer_id, article_id
        FROM t_recent
        INNER JOIN sampled_users USING(customer_id)
    ),

    -- 5) 인기 pool(negative 후보 아이템 pool) 크게 잡기
    pop_pool AS (
        SELECT article_id::VARCHAR AS article_id
        FROM read_parquet('data/features/item_features.parquet')
        ORDER BY popularity_rank ASC, article_id ASC
        LIMIT {int(popularity_pool)}
    ),

    -- 6) 유저별 negative 샘플: pop_pool에서 구매이력 제외 후 유저당 N개
    user_neg AS (
        SELECT customer_id, article_id, 0 AS label
        FROM (
            SELECT
                su.customer_id,
                pp.article_id,
                ROW_NUMBER() OVER (
                    PARTITION BY su.customer_id
                    ORDER BY random()
                ) AS rn
            FROM sampled_users su
            CROSS JOIN pop_pool pp
            LEFT JOIN user_purchased up
                ON up.customer_id = su.customer_id
               AND up.article_id  = pp.article_id
            WHERE up.article_id IS NULL
        )
        WHERE rn <= {int(effective_negative)}
    ),

    all_samples AS (
        SELECT * FROM user_pos
        UNION ALL
        SELECT * FROM user_neg
    )

    SELECT
        s.customer_id,
        s.article_id,
        s.label,

        -- user features
        uf.purchase_count,
        uf.recency,
        uf.purchase_frequency,

        -- item features
        it.popularity_rank,
        it.sales_count

    FROM all_samples s
    INNER JOIN read_parquet('data/features/user_features.parquet') uf
        ON s.customer_id = uf.customer_id
    INNER JOIN read_parquet('data/features/item_features.parquet') it
        ON s.article_id = it.article_id
    ORDER BY s.customer_id ASC, s.label DESC, s.article_id ASC
    """

    logger.info("SQL 실행 중...")
    pdf = con.execute(query).fetch_df()
    con.close()

    df = pl.from_pandas(pdf)

    # 최소 sanity check
    pos = df.filter(pl.col("label") == 1).height
    neg = df.filter(pl.col("label") == 0).height
    logger.info(f"샘플 수: {df.height:,} (pos={pos:,}, neg={neg:,})")

    # 유저당 정확히 1 pos인지 확인 (깨지면 데이터가 꼬인 것)
    per_user_pos = (
        df.filter(pl.col("label") == 1)
          .group_by("customer_id")
          .len()
          .select(pl.col("len").min().alias("min_pos"), pl.col("len").max().alias("max_pos"))
    )
    logger.info(f"per-user pos count(min/max): {per_user_pos.row(0)} (target={pos_per_user})")

    # Encode categorical features
    # purchase_frequency is a string column (low/medium/high), need to convert to numeric
    df = df.with_columns([
        pl.when(pl.col("purchase_frequency") == "low").then(0)
          .when(pl.col("purchase_frequency") == "medium").then(1)
          .when(pl.col("purchase_frequency") == "high").then(2)
          .otherwise(0)  # default to low if unknown
          .alias("purchase_frequency_encoded")
    ])
    
    feature_cols = [
        "purchase_count", "recency", "purchase_frequency_encoded",
        "popularity_rank", "sales_count"
    ]
    features = df.select(feature_cols)
    labels = df.select("label")

    # group 생성: customer_id별 row 수
    group_df = df.group_by("customer_id").len().sort("customer_id")
    group = group_df["len"].to_list()

    # group과 row수 일치 체크
    if sum(group) != df.height:
        raise ValueError(f"group 합({sum(group)}) != rows({df.height})")

    return features, labels, group


if __name__ == "__main__":
    X, y, group = create_ranking_dataset(
        sample_users=5000,
        pos_per_user=5, 
        negative_per_user=3,
        window_days=60,
        popularity_pool=1000,
        seed=42,
    )

    logger.info(f"Features: {X.shape}, Labels: {y.shape}, #groups={len(group)}")
    logger.info(f"Label ratio: pos={(y.filter(pl.col('label')==1).height / y.height):.4f}")
    logger.info(
        f"pos_per_user={pos_per_user}, "
        f"negative_per_user={negative_per_user}, "
        f"effective_negative={effective_negative}"
    )