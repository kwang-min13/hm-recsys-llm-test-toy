import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.ab_test import ABTestSimulator
from src.models.serving import RecommendationService
from src.models.candidate_generation import CandidateGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyAB")

def verify_ab_simulation():
    # Initialize components
    logger.info("Initializing components...")
    gen = CandidateGenerator(
        db_path="local_helix.db", 
        cf_window_days=60,
        articles_path="data/articles.csv"
    )
    rec_service = RecommendationService(candidate_k=100) # Uses default gen internally but we only need it for B
    
    # We replace the internal generator of rec_service to share the one we configured if needed, 
    # but actually rec_service instantiates its own. 
    # Let's trust rec_service's own instantiation since we updated the class definition default.
    
    simulator = ABTestSimulator(
        ollama_client=None,
        rec_service=rec_service,
        candidate_gen=gen, # For Group A
        mode='fast'
    )
    
    # Test User (Sparse user from previous step)
    user_id = "a9119a5febb984b34a6401e8872cba13e83d8e6db25d51d9715cc1184ea9eab8"
    
    try:
        logger.info(f"Simulating Group B for user: {user_id}")
        result = simulator.simulate_group_b(user_id)
        
        strategy = result.get('strategy')
        segment = result.get('segment')
        items = result.get('items')
        
        logger.info(f"Result Strategy: {strategy}")
        logger.info(f"Result Segment: {segment}")
        logger.info(f"Num Items: {len(items)}")
        
        if strategy == "enhanced_fallback":
            logger.info("SUCCESS: User correctly routed to Enhanced Fallback (Case B).")
        else:
            logger.error(f"FAILURE: Expected 'enhanced_fallback', got '{strategy}'")
            
    finally:
        simulator.close()

if __name__ == "__main__":
    verify_ab_simulation()
