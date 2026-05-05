"""
Test Group B Recommendation Generation

Group B (ML 모델) 추천 생성 테스트
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.serving import RecommendationService
from src.models.candidate_generation import CandidateGenerator
import duckdb

print("=" * 80)
print("GROUP B RECOMMENDATION TEST")
print("=" * 80)

# 1. 샘플 유저 가져오기
print("\n[1/4] Getting sample user...")
print("-" * 80)

con = duckdb.connect(':memory:')
try:
    sample_user = con.execute("""
        SELECT customer_id 
        FROM read_parquet('data/features/user_features.parquet')
        LIMIT 1
    """).fetchone()[0]
    print(f"Sample user: {sample_user}")
except Exception as e:
    print(f"Error getting sample user: {e}")
    exit(1)
finally:
    con.close()

# 2. RecommendationService 초기화
print("\n[2/4] Initializing RecommendationService...")
print("-" * 80)

try:
    rec_service = RecommendationService()
    print("RecommendationService initialized successfully")
except Exception as e:
    print(f"Error initializing RecommendationService: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 3. 추천 생성 테스트
print("\n[3/4] Generating recommendations...")
print("-" * 80)

try:
    recommendations = rec_service.recommend(sample_user, top_k=5)
    print(f"Recommendations generated: {type(recommendations)}")
    print(f"Keys: {recommendations.keys() if isinstance(recommendations, dict) else 'N/A'}")
    
    if isinstance(recommendations, dict):
        rec_items = recommendations.get('recommendations', [])
        optimal_time = recommendations.get('optimal_send_time', 12)
        
        print(f"\nRecommendation items: {len(rec_items)}")
        print(f"Optimal send time: {optimal_time}")
        
        if rec_items:
            print(f"\nFirst 5 items:")
            for i, item in enumerate(rec_items[:5], 1):
                print(f"  {i}. {item}")
        else:
            print("\n[WARNING] Recommendation list is EMPTY!")
            print("This is the problem causing Group B issues")
    else:
        print(f"[ERROR] Unexpected return type: {type(recommendations)}")
        
except Exception as e:
    print(f"[ERROR] Failed to generate recommendations: {e}")
    import traceback
    traceback.print_exc()

# 4. CandidateGenerator 직접 테스트
print("\n[4/4] Testing CandidateGenerator directly...")
print("-" * 80)

try:
    candidate_gen = CandidateGenerator()
    
    # Popularity candidates
    print("\nTesting popularity candidates...")
    pop_candidates = candidate_gen.generate_popularity_candidates(top_k=5)
    print(f"Popularity candidates: {len(pop_candidates)}")
    if pop_candidates:
        print(f"  First 5: {pop_candidates[:5]}")
    else:
        print("  [WARNING] Empty!")
    
    # Merged candidates
    print("\nTesting merged candidates...")
    merged = candidate_gen.merge_candidates(sample_user, total_k=10)
    print(f"Merged candidates: {len(merged)}")
    if merged:
        print(f"  First 5: {merged[:5]}")
    else:
        print("  [WARNING] Empty!")
    
    candidate_gen.close()
    
except Exception as e:
    print(f"[ERROR] CandidateGenerator test failed: {e}")
    import traceback
    traceback.print_exc()

# 정리
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("\nIf recommendations are empty, possible causes:")
print("  1. DuckDB connection issue (file locking)")
print("  2. Model file not loaded properly")
print("  3. Candidate generation returning empty list")
print("  4. User has no purchase history")
print("\nCheck the output above to identify the root cause.")
print("=" * 80)

try:
    rec_service.close()
except:
    pass
