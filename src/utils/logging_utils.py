"""
Logging Utilities

이 모듈은 프로젝트 전반에 걸쳐 일관된 로깅을 제공합니다.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    로거 설정
    
    Args:
        name: 로거 이름
        log_file: 로그 파일 경로 (None이면 파일 로깅 안 함)
        level: 로그 레벨
        format_string: 로그 포맷 문자열
        
    Returns:
        설정된 Logger 객체
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()
    
    # 기본 포맷
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (옵션)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    기존 로거 가져오기 또는 새로 생성
    
    Args:
        name: 로거 이름
        
    Returns:
        Logger 객체
    """
    logger = logging.getLogger(name)
    
    # 핸들러가 없으면 기본 설정
    if not logger.handlers:
        setup_logger(name)
    
    return logger


class PerformanceLogger:
    """성능 측정 로거"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time: Optional[datetime] = None
    
    def start(self, task_name: str):
        """작업 시작"""
        self.start_time = datetime.now()
        self.logger.info(f"[START] {task_name}")
    
    def end(self, task_name: str):
        """작업 종료 및 소요 시간 로깅"""
        if self.start_time is None:
            self.logger.warning("start()가 호출되지 않았습니다.")
            return
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"[END] {task_name} - 소요 시간: {elapsed:.2f}초")
        self.start_time = None


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = ""):
    """
    메트릭 딕셔너리를 보기 좋게 로깅
    
    Args:
        logger: Logger 객체
        metrics: 메트릭 딕셔너리
        prefix: 로그 메시지 접두사
    """
    logger.info("=" * 60)
    if prefix:
        logger.info(prefix)
    
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)


def main():
    """테스트"""
    # 기본 로거
    logger = setup_logger('test_logger', log_file='logs/test.log')
    logger.info("테스트 로그 메시지")
    
    # 성능 로거
    perf_logger = PerformanceLogger(logger)
    perf_logger.start("테스트 작업")
    import time
    time.sleep(1)
    perf_logger.end("테스트 작업")
    
    # 메트릭 로깅
    metrics = {
        'ndcg@10': 0.8523,
        'hit_rate@10': 0.7234,
        'precision@10': 0.6891
    }
    log_metrics(logger, metrics, "모델 평가 결과:")


if __name__ == "__main__":
    main()
