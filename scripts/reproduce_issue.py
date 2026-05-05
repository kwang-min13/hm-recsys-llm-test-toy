
import sys
from pathlib import Path
import logging
import duckdb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.persona import load_user_metadata, load_user_metadata_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_fix():
    logger.info("Verifying fix for load_user_metadata...")
    
    con = duckdb.connect(":memory:")
    
    # Check if local_helix.db exists
    db_path = Path('local_helix.db')
    if db_path.exists():
        logger.info(f"Attaching {db_path}...")
        con.execute(f"ATTACH '{db_path}' AS filedb (READ_ONLY)")
        
        # Get a sample user
        try:
            sample_user = con.execute("SELECT customer_id FROM filedb.user_features LIMIT 1").fetchone()
            if not sample_user:
                logger.error("No users found in DB")
                return
            user_id = sample_user[0]
            logger.info(f"Testing with user_id: {user_id}")
            
            # Test single load
            logger.info("Testing load_user_metadata...")
            metadata = load_user_metadata(con, user_id)
            if metadata:
                logger.info(f"Successfully loaded metadata: {metadata}")
            else:
                logger.error("Failed to load metadata (returned None)")

            # Test batch load
            logger.info("Testing load_user_metadata_batch...")
            batch_metadata = load_user_metadata_batch(con, [user_id])
            if batch_metadata and user_id in batch_metadata:
                logger.info(f"Successfully loaded batch metadata: {batch_metadata[user_id]}")
            else:
                logger.error("Failed to load batch metadata")
                
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        logger.warning("local_helix.db not found. Trying to read from parquet directly to mock the table structure.")
        # Create a mock view if DB doesn't exist, just to test the logic if possible
        # But load_user_metadata expects 'filedb.user_features'
        # We can simulate this schema
        try:
            con.execute("CREATE SCHEMA IF NOT EXISTS filedb")
            # Create a dummy table with correct schema
            con.execute("""
                CREATE TABLE filedb.user_features (
                    customer_id VARCHAR,
                    purchase_frequency VARCHAR,
                    avg_price DOUBLE,
                    recency INTEGER
                )
            """)
            con.execute("INSERT INTO filedb.user_features VALUES ('test_user', 'high', 0.05, 3)")
            
            user_id = 'test_user'
            logger.info(f"Testing with mock user_id: {user_id}")
            
            # Test single load
            metadata = load_user_metadata(con, user_id)
            if metadata:
                logger.info(f"Successfully loaded metadata: {metadata}")
            else:
                logger.error("Failed to load metadata")

             # Test batch load
            batch_metadata = load_user_metadata_batch(con, [user_id])
            if batch_metadata:
                logger.info(f"Successfully loaded batch metadata: {batch_metadata}")
            else:
                logger.error("Failed to load batch metadata")

        except Exception as e:
             logger.error(f"Error setting up mock DB: {e}")

    con.close()

if __name__ == "__main__":
    verify_fix()
