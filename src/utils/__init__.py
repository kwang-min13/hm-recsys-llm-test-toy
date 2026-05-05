"""Utility functions for Local-Helix project"""

from .db_init import DuckDBManager, create_database_schema, test_connection
from .logging_utils import setup_logger, get_logger, PerformanceLogger, log_metrics
from .validation import (
    validate_dataframe,
    validate_user_features,
    validate_item_features,
    validate_predictions,
    validate_model_output,
    ValidationError
)

__all__ = [
    'DuckDBManager',
    'create_database_schema',
    'test_connection',
    'setup_logger',
    'get_logger',
    'PerformanceLogger',
    'log_metrics',
    'validate_dataframe',
    'validate_user_features',
    'validate_item_features',
    'validate_predictions',
    'validate_model_output',
    'ValidationError'
]
