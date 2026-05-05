"""
Candidate Generation Module (Improved v2)

- Popularity candidates (deterministic + scored)
- Item-to-item co-purchase CF candidates (time decay + user-recent weighting + popularity penalty)
- Score-based deterministic merge (robust normalization, clear tie-break)
- DuckDB: transactions scanned ONCE (TEMP TABLE materialization)

Key fixes vs v1:
1) Avoid CSV re-scan by materializing CF window into TEMP TABLE
2) Switch CF to item-to-item co-occurrence (more stable + faster than similar-users overlap)
3) Apply user recent-item weights + time decay
4) Optional popularity penalty (rank-based, but isolated & tunable)
5) Normalize pop/cf scores before weighted merge
"""

from __future__ import annotations

import duckdb
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import math
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredItem:
    item_id: str
    score: float
    source: str  # "pop" or "cf"


class CandidateGenerator:
    def __init__(
        self,
        db_path: str = "local_helix.db",
        transactions_path: str = "data/transactions_train.csv",
        item_features_path: str = "data/features/item_features.parquet",
        articles_path: str = "data/articles.csv",
        memory_limit: str = "8GB",
        threads: int = 4,
        cf_window_days: int = 28,
        materialize_transactions: bool = True,
    ):
        self.db_path = db_path
        self.transactions_path = transactions_path
        self.item_features_path = item_features_path
        self.articles_path = articles_path
        self.memory_limit = memory_limit
        self.threads = threads
        self.cf_window_days = cf_window_days
        self.materialize_transactions = materialize_transactions

        self.con: Optional[duckdb.DuckDBPyConnection] = None
        self._cache_ready = False

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self.con is None:
            # Use in-memory database to avoid file locking
            self.con = duckdb.connect(":memory:")
            # Attach the file database in read-only mode for data access
            if self.db_path != ":memory:":
                try:
                    self.con.execute(f"ATTACH '{self.db_path}' AS filedb (READ_ONLY)")
                except Exception:
                    # If attach fails, continue with in-memory only
                    pass
            self.con.execute(f"SET memory_limit='{self.memory_limit}'")
            self.con.execute(f"SET threads TO {int(self.threads)}")
        if not self._cache_ready:
            self._prepare_cache()
        return self.con

    def _prepare_cache(self) -> None:
        con = self.con
        assert con is not None

        # item features view
        con.execute("DROP VIEW IF EXISTS v_item_features")
        con.execute(
            f"""
            CREATE VIEW v_item_features AS
            SELECT
                article_id::VARCHAR AS article_id,
                popularity_rank
            FROM read_parquet('{self.item_features_path}')
            """
        )

        # articles view (for category info)
        con.execute("DROP VIEW IF EXISTS v_articles")
        con.execute(
            f"""
            CREATE VIEW v_articles AS
            SELECT
                article_id::VARCHAR AS article_id,
                product_group_name  -- Category
            FROM read_csv_auto('{self.articles_path}', header=true)
            """
        )

        # raw transactions view (will be materialized into temp table for CF window)
        con.execute("DROP VIEW IF EXISTS v_transactions_all")
        con.execute(
            f"""
            CREATE VIEW v_transactions_all AS
            SELECT
                customer_id::VARCHAR AS customer_id,
                article_id::VARCHAR  AS article_id,
                CAST(t_dat AS DATE)  AS t_dat
            FROM read_csv_auto('{self.transactions_path}', header=true)
            """
        )

        # materialize CF window
        con.execute("DROP TABLE IF EXISTS t_cf_transactions")
        con.execute("DROP VIEW IF EXISTS v_cf_transactions")

        if self.materialize_transactions:
            # IMPORTANT: force one-time scan + keep only window
            con.execute(
                f"""
                CREATE TEMP TABLE t_cf_transactions AS
                WITH mx AS (SELECT MAX(t_dat) AS dmax FROM v_transactions_all)
                SELECT *
                FROM v_transactions_all
                WHERE t_dat >= (SELECT dmax - INTERVAL '{int(self.cf_window_days)} days' FROM mx)
                """
            )
            # Helpful: order by (article_id, customer_id, t_dat) for join locality
            con.execute("DROP TABLE IF EXISTS t_cf_sorted")
            con.execute(
                """
                CREATE TEMP TABLE t_cf_sorted AS
                SELECT * FROM t_cf_transactions
                ORDER BY article_id, customer_id, t_dat
                """
            )
            con.execute("DROP TABLE IF EXISTS t_cf_transactions")
            con.execute("ALTER TABLE t_cf_sorted RENAME TO t_cf_transactions")

            con.execute("CREATE VIEW v_cf_transactions AS SELECT * FROM t_cf_transactions")
        else:
            con.execute(
                f"""
                CREATE VIEW v_cf_transactions AS
                WITH mx AS (SELECT MAX(t_dat) AS dmax FROM v_transactions_all)
                SELECT *
                FROM v_transactions_all
                WHERE t_dat >= (SELECT dmax - INTERVAL '{int(self.cf_window_days)} days' FROM mx)
                """
            )

        # cache max date for consistent time-decay
        con.execute("DROP VIEW IF EXISTS v_cf_max_date")
        con.execute(
            """
            CREATE VIEW v_cf_max_date AS
            SELECT MAX(t_dat) AS dmax
            FROM v_cf_transactions
            """
        )

        self._cache_ready = True
        logger.info("Cache ready: CF window materialized=%s", self.materialize_transactions)

    # ---------------------------
    # Popularity
    # ---------------------------
    def generate_popularity_candidates(self, top_k: int = 50) -> List[str]:
        con = self.connect()
        q = """
            SELECT article_id
            FROM v_item_features
            ORDER BY popularity_rank ASC NULLS LAST, article_id ASC
            LIMIT ?
        """
        rows = con.execute(q, [int(top_k)]).fetchall()
        return [r[0] for r in rows]

    def generate_popularity_scored(self, top_k: int = 50) -> List[ScoredItem]:
        """
        score_pop = 1 / (1 + popularity_rank)
        """
        con = self.connect()
        q = """
            SELECT article_id, popularity_rank
            FROM v_item_features
            ORDER BY popularity_rank ASC NULLS LAST, article_id ASC
            LIMIT ?
        """
        rows = con.execute(q, [int(top_k)]).fetchall()

        out: List[ScoredItem] = []
        for item_id, pop_rank in rows:
            r = float(pop_rank) if pop_rank is not None else 1e12
            r = max(r, 0.0)
            score = 1.0 / (1.0 + r)
            out.append(ScoredItem(item_id=str(item_id), score=float(score), source="pop"))
        return out

    # ---------------------------
    # Category (New for Case B)
    # ---------------------------
    def generate_category_scored(self, category: str, top_k: int = 50) -> List[ScoredItem]:
        """
        Fetch items from a specific category, scored by global popularity.
        """
        con = self.connect()
        # Join v_item_features (popularity) with v_articles (category)
        q = """
            SELECT
                f.article_id,
                f.popularity_rank
            FROM v_item_features f
            JOIN v_articles a ON f.article_id = a.article_id
            WHERE a.product_group_name = ?
            ORDER BY f.popularity_rank ASC NULLS LAST, f.article_id ASC
            LIMIT ?
        """
        rows = con.execute(q, [category, int(top_k)]).fetchall()

        out: List[ScoredItem] = []
        for item_id, pop_rank in rows:
            r = float(pop_rank) if pop_rank is not None else 1e12
            r = max(r, 0.0)
            score = 1.0 / (1.0 + r)
            out.append(ScoredItem(item_id=str(item_id), score=float(score), source="category"))
        return out

    # ---------------------------
    # CF: item-to-item co-purchase
    # ---------------------------
    
    def generate_cf_scored_item2item(
        self,
        user_id: str,
        top_k: int = 50,
        recent_items: int = 10,
        cooc_top_per_seed: int = 200,
        time_decay_half_life_days: int = 14,
        popularity_penalty_alpha: float = 0.20,
        exclude_already_purchased: bool = True,
        # ✅ 추가: co-occurrence 제약
        cooc_same_day_only: bool = True,         # 기본: 같은 날만 cooc로 인정
        cooc_max_day_gap: int = 3,               # same_day_only=False일 때 ±N일 (0이면 same-day와 동일)
    ) -> List[ScoredItem]:
        """
        CF score intuition:
        - Take user's recent items (seed set)
        - For each seed item, find co-purchased items in same window
        - Weight by:
            (a) seed recency weight
            (b) time decay on co-purchase transactions
        - Apply popularity penalty optionally
        """
        con = self.connect()
        half_life = max(int(time_decay_half_life_days), 1)

        # ✅ cooc 제약 조건 SQL 조각
        # - same_day_only=True  -> t2.t_dat = t1.t_dat
        # - same_day_only=False -> |t2.t_dat - t1.t_dat| <= cooc_max_day_gap
        if cooc_same_day_only:
            cooc_date_cond = "AND t2.t_dat = t1.t_dat"
        else:
            gap = max(int(cooc_max_day_gap), 0)
            cooc_date_cond = f"AND ABS(DATE_DIFF('day', t2.t_dat, t1.t_dat)) <= {gap}"

        q = f"""
        WITH
        dmax AS (SELECT dmax FROM v_cf_max_date),
        user_recent AS (
            SELECT
                article_id AS seed_item,
                t_dat      AS seed_date,
                ROW_NUMBER() OVER (ORDER BY t_dat DESC, article_id ASC) AS rnk
            FROM v_cf_transactions
            WHERE customer_id = ?
        QUALIFY rnk <= ?
        ),
        seed_weighted AS (
            SELECT
                seed_item,
                seed_date,
                POW(
                0.5,
                DATE_DIFF('day', seed_date, (SELECT dmax FROM dmax))::DOUBLE / {half_life}.0
            ) AS w_seed
        FROM user_recent
        ),
        user_purchased AS (
            SELECT DISTINCT article_id
            FROM v_cf_transactions
            WHERE customer_id = ?
        ),
        cooc_raw AS (
            SELECT
                sw.seed_item,
                t2.article_id AS cand_item,
            SUM(
                sw.w_seed
                * POW(
                    0.5,
                    DATE_DIFF('day', t2.t_dat, (SELECT dmax FROM dmax))::DOUBLE / {half_life}.0
                )
            ) AS raw_score
        FROM seed_weighted sw
        JOIN v_cf_transactions t1
          ON t1.article_id = sw.seed_item
        JOIN v_cf_transactions t2
          ON t2.customer_id = t1.customer_id
        WHERE t2.article_id <> sw.seed_item
          {cooc_date_cond}   -- ✅ 핵심: same-day/근접일 cooc 제약
        GROUP BY sw.seed_item, t2.article_id
        ),
        cooc_pruned AS (
            SELECT *
            FROM (
                SELECT
                    seed_item,
                    cand_item,
                raw_score,
                ROW_NUMBER() OVER (PARTITION BY seed_item ORDER BY raw_score DESC, cand_item ASC) AS rr
            FROM cooc_raw
        )
        WHERE rr <= ?
        ),
        cand_agg AS (
            SELECT
                cand_item AS article_id,
                SUM(raw_score) AS score_sum
            FROM cooc_pruned
            GROUP BY cand_item
        ),
        cand_join AS (
            SELECT
                ca.article_id,
                ca.score_sum,
                vf.popularity_rank
            FROM cand_agg ca
            LEFT JOIN v_item_features vf
              ON vf.article_id = ca.article_id
        ),
        cand_filtered AS (
            SELECT
                article_id,
                CASE
                    WHEN popularity_rank IS NULL THEN score_sum
                    ELSE score_sum / (1.0 + ? * LN(1.0 + CAST(popularity_rank AS DOUBLE)))
                END AS score_cf
            FROM cand_join
            {"WHERE article_id NOT IN (SELECT article_id FROM user_purchased)" if exclude_already_purchased else ""}
         )
        SELECT article_id, score_cf
        FROM cand_filtered
        ORDER BY score_cf DESC, article_id ASC
        LIMIT ?
        """

        params = [
            str(user_id),
            int(recent_items),
            str(user_id),
            int(cooc_top_per_seed),
            float(popularity_penalty_alpha),
            int(top_k),
        ]

        rows = con.execute(q, params).fetchall()
        return [ScoredItem(item_id=str(i), score=float(s), source="cf") for i, s in rows]


    # ---------------------------
    # Merge: robust normalization + deterministic ranking
    # ---------------------------
    @staticmethod
    def _rrf_scores(ordered_ids: List[str], k: int = 60) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion score.
        score = 1 / (k + rank) , rank starts at 1
        """
        out: Dict[str, float] = {}
        for idx, item_id in enumerate(ordered_ids, start=1):
            out[item_id] = 1.0 / (float(k) + float(idx))
        return out

    def merge_candidates(
        self,
        user_id: str,
        total_k: int = 100,
        pop_top: int = 200,
        cf_top: int = 300,
        w_pop: float = 0.15,
        w_cf: float = 0.70,
        recent_items: int = 10,
        cooc_top_per_seed: int = 200,
        time_decay_half_life_days: int = 14,
        popularity_penalty_alpha: float = 0.20,
        fallback_pop_expand: int = 1000,
        # ✅ RRF 하이퍼파라미터(클수록 상위편향 약해짐)
        rrf_k: int = 60,
        # ✅ CF cooc 제약 옵션도 merge에서 노출
        cooc_same_day_only: bool = True,
        cooc_max_day_gap: int = 0,
    ) -> List[str]:
        # ---- DEBUG: 경로/존재 확인 ----
        txp = Path(self.transactions_path)
        itp = Path(self.item_features_path)
        logger.warning("[DEBUG] cwd=%s", Path.cwd())
        logger.warning("[DEBUG] transactions_path=%s exists=%s", txp, txp.exists())
        logger.warning("[DEBUG] item_features_path=%s exists=%s", itp, itp.exists())

        con = self.connect()

        # ---- DEBUG: 실제 읽힌 row 수 확인 ----
        try:
            n_item = con.execute("SELECT COUNT(*) FROM v_item_features").fetchone()[0]
            n_cf = con.execute("SELECT COUNT(*) FROM v_cf_transactions").fetchone()[0]
            logger.warning("[DEBUG] v_item_features rows=%s, v_cf_transactions rows=%s", n_item, n_cf)
        except Exception as e:
            logger.error("[DEBUG] view count failed: %s", e)
            
        total_k = int(total_k)
        if total_k <= 0:
            return []

        pop_scored = self.generate_popularity_scored(top_k=int(pop_top))
        cf_scored = self.generate_cf_scored_item2item(
            user_id=user_id,
            top_k=int(cf_top),
            recent_items=int(recent_items),
            cooc_top_per_seed=int(cooc_top_per_seed),
            time_decay_half_life_days=int(time_decay_half_life_days),
            popularity_penalty_alpha=float(popularity_penalty_alpha),
            cooc_same_day_only=bool(cooc_same_day_only),
            cooc_max_day_gap=int(cooc_max_day_gap),
        )
        logger.warning("[DEBUG] pop_scored=%d, cf_scored=%d", len(pop_scored), len(cf_scored))
        # ✅ 점수 기반 정렬(동점은 item_id로 deterministic)
        pop_ordered = [it.item_id for it in sorted(pop_scored, key=lambda x: (-x.score, x.item_id))]
        cf_ordered = [it.item_id for it in sorted(cf_scored, key=lambda x: (-x.score, x.item_id))]

        pop_rrf = self._rrf_scores(pop_ordered, k=int(rrf_k))
        cf_rrf = self._rrf_scores(cf_ordered, k=int(rrf_k))

        all_ids = set(pop_rrf.keys()) | set(cf_rrf.keys())

    # ✅ 부족하면 pop 확장 (기존과 동일하지만, RRF 점수로만 들어옴)
        if len(all_ids) < total_k:
            expanded = self.generate_popularity_scored(top_k=int(fallback_pop_expand))
            exp_ordered = [it.item_id for it in sorted(expanded, key=lambda x: (-x.score, x.item_id))]
            exp_rrf = self._rrf_scores(exp_ordered, k=int(rrf_k))
            for item_id in exp_ordered:
                if len(all_ids) >= total_k:
                    break
                if item_id not in all_ids:
                    pop_rrf[item_id] = exp_rrf.get(item_id, 0.0)
                    all_ids.add(item_id)

    # ✅ 최종 결합 점수
        def key_fn(item_id: str):
            ps = float(pop_rrf.get(item_id, 0.0))
            cs = float(cf_rrf.get(item_id, 0.0))
            final = float(w_pop) * ps + float(w_cf) * cs
            # tie-break: final -> cf -> pop -> item_id (deterministic)
            return (final, cs, ps, item_id)


        ranked = sorted(all_ids, key=key_fn, reverse=True)
        return ranked[:total_k]

    # ---------------------------
    # Case B: Low-Activity Users (Popularity + Recency)
    # ---------------------------
    
    def generate_popularity_with_recency(
        self,
        top_k: int = 100,
        recency_window_days: int = 30,
        w_popularity: float = 0.6,
        w_recency: float = 0.4,
    ) -> List[ScoredItem]:
        """
        Generate candidates for low-activity users using popularity + recency.
        
        Args:
            top_k: Number of candidates to return
            recency_window_days: Window for calculating recent trending items
            w_popularity: Weight for popularity score
            w_recency: Weight for recency score
            
        Returns:
            List of scored items (popularity + recency combined)
        """
        con = self.connect()
        
        # Get popularity scores
        pop_scored = self.generate_popularity_scored(top_k=int(top_k * 2))  # Get more for merging
        
        # Get recency scores (items with high recent sales velocity)
        q = f"""
        WITH recent_sales AS (
            SELECT
                article_id,
                COUNT(*) AS recent_count
            FROM v_cf_transactions
            WHERE t_dat >= (SELECT dmax - INTERVAL '{int(recency_window_days)} days' FROM v_cf_max_date)
            GROUP BY article_id
        ),
        recency_scored AS (
            SELECT
                article_id,
                recent_count,
                recent_count::DOUBLE / (1.0 + (SELECT MAX(recent_count) FROM recent_sales)) AS recency_score
            FROM recent_sales
        )
        SELECT article_id, recency_score
        FROM recency_scored
        ORDER BY recency_score DESC, article_id ASC
        LIMIT ?
        """
        
        recency_rows = con.execute(q, [int(top_k * 2)]).fetchall()
        recency_dict = {str(item_id): float(score) for item_id, score in recency_rows}
        
        # Combine scores
        combined: Dict[str, float] = {}
        all_items = set([it.item_id for it in pop_scored]) | set(recency_dict.keys())
        
        for item_id in all_items:
            pop_score = next((it.score for it in pop_scored if it.item_id == item_id), 0.0)
            rec_score = recency_dict.get(item_id, 0.0)
            combined[item_id] = float(w_popularity) * pop_score + float(w_recency) * rec_score
        
        # Sort and return top_k
        sorted_items = sorted(combined.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        
        return [
            ScoredItem(item_id=item_id, score=score, source="pop_recency")
            for item_id, score in sorted_items
        ]
    
    def _get_user_preferred_category(self, user_id: str, limit: int = 10) -> Optional[str]:
        """
        Find the most frequent category in user's recent history.
        """
        con = self.connect()
        q = """
            WITH recent AS (
                SELECT article_id
                FROM v_cf_transactions
                WHERE customer_id = ?
                ORDER BY t_dat DESC
                LIMIT ?
            )
            SELECT
                a.product_group_name,
                COUNT(*) as cnt
            FROM recent r
            JOIN v_articles a ON r.article_id = a.article_id
            GROUP BY a.product_group_name
            ORDER BY cnt DESC
            LIMIT 1
        """
        row = con.execute(q, [user_id, int(limit)]).fetchone()
        return row[0] if row else None

    def merge_candidates_case_b(
        self,
        user_id: str,
        total_k: int = 100,
        recency_window_days: int = 30,
        w_popularity: float = 0.5,
        w_recency: float = 0.2,
        w_category: float = 0.3,  # Added weight for category
    ) -> List[str]:
        """
        Generate candidates for Case B (low-activity users).
        Strategty: Popularity + Recency + Category Fallback (if available).
        
        Args:
            user_id: User ID
            total_k: Number of candidates to return
            recency_window_days: Window for recency calculation
            w_popularity: Weight for popularity
            w_recency: Weight for recency
            w_category: Weight for category fallback
            
        Returns:
            List of item IDs (deterministic order)
        """
        total_k = int(total_k)
        if total_k <= 0:
            return []
        
        # 1. Get Popularity + Recency candidates
        # (We allocate full total_k for them first as base)
        pop_rec_scored = self.generate_popularity_with_recency(
            top_k=int(total_k * 1.5),
            recency_window_days=recency_window_days,
            w_popularity=w_popularity,  # Note: Weights might need re-balancing if category is added
            w_recency=w_recency,
        )
        
        # 2. Try to find user's preferred category
        pref_category = self._get_user_preferred_category(user_id)
        cat_scored: List[ScoredItem] = []
        
        if pref_category:
            # Fetch popular items from this category
            cat_scored = self.generate_category_scored(pref_category, top_k=total_k)
            logger.info(f"User {user_id}: Found preferred category '{pref_category}', added {len(cat_scored)} candidates.")
        
        # 3. Merge Strategies
        # We will use a simple weighted additive score again.
        # Normalize scores to 0-1 range roughly effectively happens via rank or raw score nature.
        # Here we reuse the existing scores which are somewhat normalized (1/(1+rank)).
        
        combined: Dict[str, float] = {}
        
        # Helper to get score from list
        def get_score(items: List[ScoredItem], iid: str) -> float:
            return next((it.score for it in items if it.item_id == iid), 0.0)

        all_ids = set([it.item_id for it in pop_rec_scored]) | set([it.item_id for it in cat_scored])
        
        for item_id in all_ids:
            # pop_rec_scored already has w_pop and w_rec applied relative to each other.
            # But let's assume we treat pop_rec_scored result as one "source".
            # Actually, to be precise, we should re-mix.
            # But for simplicity, let's treat the Output of generate_popularity_with_recency as "Global Trend Score".
            
            s_trend = get_score(pop_rec_scored, item_id)
            s_cat = get_score(cat_scored, item_id)
            
            # Final Score: Trend + Category
            # Trend score already includes pop(0.6) + rec(0.4) roughly.
            # We want to boost if it matches category.
            
            final_score = s_trend + (s_cat * w_category)
            combined[item_id] = final_score

        # Sort
        sorted_items = sorted(combined.items(), key=lambda x: (-x[1], x[0]))[:total_k]
        
        return [item_id for item_id, _ in sorted_items]


    def close(self) -> None:
        if self.con is not None:
            self.con.close()
            self.con = None
        self._cache_ready = False


def main():
    import time

    gen = CandidateGenerator(
        db_path="local_helix.db",
        transactions_path="data/transactions_train.csv",
        item_features_path="data/features/item_features.parquet",
        cf_window_days=56,
        materialize_transactions=True,
    )

    try:
        con = gen.connect()
        sample_user = con.execute(
            """
            SELECT customer_id
            FROM read_parquet('data/features/user_features.parquet')
            LIMIT 1
            """
        ).fetchone()[0]

        logger.info("Sample user: %s", sample_user)

        t0 = time.time()
        pop = gen.generate_popularity_candidates(top_k=50)
        logger.info("Popularity: %d (%.1f ms)", len(pop), (time.time() - t0) * 1000)

        t0 = time.time()
        cf = gen.generate_cf_scored_item2item(sample_user, top_k=50)
        logger.info("CF(item2item): %d (%.1f ms)", len(cf), (time.time() - t0) * 1000)

        t0 = time.time()
        merged = gen.merge_candidates(sample_user, total_k=100)
        logger.info("Merged: %d (%.1f ms)", len(merged), (time.time() - t0) * 1000)
        logger.info("Top-10 merged: %s", merged[:10])

    finally:
        gen.close()


if __name__ == "__main__":
    main()
