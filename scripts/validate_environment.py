"""
Environment Validation Script

환경 설정 및 데이터 검증
"""

import sys
import importlib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Python 버전 확인"""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error("Python 3.10+ required")
        return False
    
    logger.info("✓ Python version OK")
    return True


def check_required_packages():
    """필수 패키지 확인"""
    required_packages = {
        'duckdb': '0.9.0',
        'polars': '0.20.0',
        'lightgbm': '4.0.0',
        'mlflow': '2.10.0',
        'streamlit': '1.30.0',
        'pandas': '2.0.0',
        'scipy': '1.11.0',
        'sklearn': '1.3.0',
    }
    
    all_ok = True
    for package, min_version in required_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            logger.info(f"✓ {package}: {version}")
        except ImportError:
            logger.error(f"✗ {package} not installed (required: {min_version}+)")
            all_ok = False
    
    return all_ok


def check_duckdb_connection():
    """DuckDB 연결 테스트"""
    try:
        import duckdb
        con = duckdb.connect(':memory:')
        result = con.execute("SELECT 'Hello DuckDB' as message").fetchone()
        con.close()
        
        logger.info(f"✓ DuckDB connection OK: {result[0]}")
        return True
    except Exception as e:
        logger.error(f"✗ DuckDB connection failed: {e}")
        return False


def check_data_files():
    """데이터 파일 존재 확인"""
    data_dir = Path('data')
    
    required_files = [
        'transactions_train.csv',
        'articles.csv',
        'customers.csv'
    ]
    
    all_ok = True
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"✓ {filename}: {size_mb:.2f} MB")
        else:
            logger.error(f"✗ {filename} not found")
            all_ok = False
    
    return all_ok


def check_feature_files():
    """Feature 파일 존재 확인"""
    feature_dir = Path('data/features')
    
    if not feature_dir.exists():
        logger.warning(f"Feature directory not found: {feature_dir}")
        logger.info("  → Features will be created in Phase 2")
        return True
    
    feature_files = [
        'user_features.parquet',
        'item_features.parquet'
    ]
    
    for filename in feature_files:
        filepath = feature_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"✓ {filename}: {size_mb:.2f} MB")
        else:
            logger.info(f"  {filename} not found (will be created)")
    
    return True


def check_model_directory():
    """모델 디렉토리 확인"""
    model_dir = Path('models/artifacts')
    
    if not model_dir.exists():
        logger.info(f"Creating model directory: {model_dir}")
        model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✓ Model directory ready: {model_dir}")
    return True


def main():
    """전체 환경 검증"""
    logger.info("=" * 60)
    logger.info("Local_Helix Environment Validation")
    logger.info("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("DuckDB Connection", check_duckdb_connection),
        ("Data Files", check_data_files),
        ("Feature Files", check_feature_files),
        ("Model Directory", check_model_directory),
    ]
    
    results = []
    for name, check_func in checks:
        logger.info(f"\n[{name}]")
        result = check_func()
        results.append((name, result))
    
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
        if not result:
            all_passed = False
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("✓ All checks passed! Ready to proceed.")
        return 0
    else:
        logger.error("✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
