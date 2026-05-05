import duckdb
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.candidate_generation import CandidateGenerator

def find_sparse_user():
    con = duckdb.connect()
    con.execute("CREATE VIEW v_tx AS SELECT * FROM read_csv_auto('data/transactions_train.csv')")
    con.execute("CREATE VIEW v_art AS SELECT * FROM read_csv_auto('data/articles.csv')")
    
    # Find a user with 2-5 purchases mostly in one category
    query = """
        SELECT 
            t.customer_id, 
            COUNT(*) as cnt, 
            list(a.product_group_name) as cats
        FROM v_tx t 
        JOIN v_art a ON t.article_id = a.article_id 
        WHERE t.t_dat > (SELECT MAX(t_dat) - INTERVAL '60 days' FROM v_tx)
        GROUP BY t.customer_id 
        HAVING cnt BETWEEN 2 AND 10
        LIMIT 5
    """
    users = con.execute(query).fetchall()
    con.close()
    return users[0][0] if users else None

def test_case_b():
    user_id = find_sparse_user()
    if not user_id:
        print("No suitable user found for testing.")
        return

    print(f"Testing with User: {user_id}")
    
    gen = CandidateGenerator(
        db_path="local_helix.db",
        transactions_path="data/transactions_train.csv",
        item_features_path="data/features/item_features.parquet",
        articles_path="data/articles.csv",
        cf_window_days=60
    )
    
    try:
        # Check preferred category
        pref_cat = gen._get_user_preferred_category(user_id)
        print(f"Preferred Category: {pref_cat}")
        
        # Generate Case B candidates
        candidates = gen.merge_candidates_case_b(
            user_id=user_id,
            total_k=20,
            w_popularity=0.3,
            w_recency=0.2,
            w_category=0.5
        )
        
        print(f"Generated {len(candidates)} candidates.")
        
        # Verify if candidates belong to the preferred category
        con = gen.connect()
        placeholders = ','.join(['?'] * len(candidates))
        q = f"""
            SELECT product_group_name, count(*) as cnt
            FROM v_articles
            WHERE article_id IN ({placeholders})
            GROUP BY product_group_name
            ORDER BY cnt DESC
        """
        stats = con.execute(q, candidates).fetchall()
        print("\nCandidate Category Distribution:")
        for cat, cnt in stats:
            print(f"- {cat}: {cnt}")
            
    finally:
        gen.close()

if __name__ == "__main__":
    test_case_b()
