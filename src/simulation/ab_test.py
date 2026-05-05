"""
A/B Test Simulator Module

A/B 테스트 시뮬레이션을 위한 모듈
- Group A (Control): 인기 상품
- Group B (Test): ML 추천

두 가지 모드 지원:
- FAST 모드: 룰 베이스만 사용 (LLM 없이 빠른 시뮬레이션)
- LLM 모드: 페르소나 기반 LLM 평가 (더 현실적인 시뮬레이션)
"""

import random
from typing import Dict, Any, Optional, Literal
import logging
import duckdb

from .virtual_user import VirtualUser
from .ollama_client import OllamaClient
from .persona import create_persona, load_user_metadata, UserMetadata
from ..models.serving import RecommendationService
from ..models.candidate_generation import CandidateGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ABTestSimulator:
    """A/B 테스트 시뮬레이터 (FAST/LLM 모드 지원)"""

    def __init__(self, 
                 ollama_client: Optional[OllamaClient],
                 rec_service: RecommendationService,
                 candidate_gen: CandidateGenerator,
                 db_path: str = 'local_helix.db',
                 mode: Literal['fast', 'llm'] = 'fast'):
        """
        초기화
        
        Args:
            ollama_client: Ollama 클라이언트 (LLM 모드에서만 필요)
            rec_service: 추천 서비스
            candidate_gen: 후보군 생성기
            db_path: DuckDB 경로
            mode: 'fast' (룰 베이스) 또는 'llm' (페르소나 기반)
        """
        self.ollama_client = ollama_client
        self.rec_service = rec_service
        self.candidate_gen = candidate_gen
        self.db_path = db_path
        self.mode = mode
        
        # LLM 모드인데 ollama_client가 없으면 경고
        if mode == 'llm' and ollama_client is None:
            logger.warning("LLM 모드이지만 ollama_client가 None입니다. FAST 모드로 전환합니다.")
            self.mode = 'fast'
        
        logger.info(f"A/B Test Simulator 초기화 완료 (모드: {self.mode.upper()})")
    def _connect_ro(self):
        con = duckdb.connect(":memory:")
        con.execute("SET threads TO 4")
        con.execute("SET memory_limit='8GB'")
        # 파일 DB는 attach로만, read-only
        con.execute(f"ATTACH '{self.db_path}' AS filedb (READ_ONLY)")
        return con  
    def _evaluate_with_llm(self, 
                          user_id: str, 
                          items: list, 
                          user_metadata: Optional[UserMetadata] = None) -> bool:
        """
        LLM 기반 평가 (페르소나 사용)
        
        Args:
            user_id: 유저 ID
            items: 추천 상품 리스트
            user_metadata: 유저 메타데이터 (없으면 DB에서 로드)
            
        Returns:
            클릭 여부
        """
        if self.ollama_client is None:
            logger.warning("LLM 클라이언트가 없어 FAST 모드로 평가합니다.")
            return self._evaluate_fast(items)
        
        # 유저 메타데이터 로드
        if user_metadata is None:
            con = self._connect_ro()
            user_metadata = load_user_metadata(con, user_id)
            con.close()
            
            if user_metadata is None:
                logger.warning(f"유저 {user_id} 메타데이터를 찾을 수 없어 FAST 모드로 평가합니다.")
                return self._evaluate_fast(items)
        
        # 페르소나 생성
        persona_prompt = create_persona(user_metadata)
        
        # LLM 프롬프트 구성
        items_str = ', '.join([str(item) for item in items[:10]])
        
        full_prompt = f"""{persona_prompt}

다음 상품 추천 푸시 알림을 받았습니다:
{items_str}

이 푸시 알림을 클릭하시겠습니까?
당신의 쇼핑 성향과 최근 구매 패턴을 고려하여 솔직하게 답변해주세요.

답변은 반드시 'Yes' 또는 'No'로만 해주세요."""

        try:
            response = self.ollama_client.generate(full_prompt)

            # ✅ response 타입 정규화 (dict or str 모두 처리)
            if isinstance(response, dict):
                response_text = str(response.get("response", "")).strip().lower()
            else:
                # response가 str(또는 다른 타입)이면 그냥 문자열로 취급
                response_text = str(response).strip().lower()

            # 응답 파싱 (Yes/No만 받는다고 했으니 더 엄격하게)
            if response_text.startswith("yes"):
                clicked = True
            elif response_text.startswith("no"):
                clicked = False
            else:
                # 애매한 응답이면 보수적으로 FAST로
                logger.warning(f"LLM 응답이 Yes/No 형식이 아님: {response_text!r} -> FAST로 대체")
                return self._evaluate_fast(items)

            logger.debug(f"LLM 평가 - User: {user_id}, Clicked: {clicked}")
            return clicked

        except Exception as e:
            logger.error(f"LLM 평가 실패: {str(e)}, FAST 모드로 대체")
            return self._evaluate_fast(items)
    
    def _evaluate_fast(self, items: list) -> bool:
        """
        FAST 모드 평가 (룰 베이스)
        
        Args:
            items: 추천 상품 리스트
            
        Returns:
            클릭 여부 (확률 기반)
        """
        # 간단한 룰: 상품이 많을수록 클릭 확률 증가
        if not items:
            return False
        
        # 기본 클릭률 30% + 상품 1개당 10% 추가 (최대 80%)
        base_ctr = 0.3
        item_bonus = min(len(items) * 0.1, 0.5)
        click_probability = base_ctr + item_bonus
        
        return random.random() < click_probability
    
    def simulate_group_a(self, user_id: str, user_metadata: Optional[UserMetadata] = None) -> Dict[str, Any]:
        """
        Group A (Control) 시뮬레이션: 인기 상품 추천
        
        Args:
            user_id: 유저 ID
            user_metadata: 유저 메타데이터 (LLM 모드에서 사용, 옵션)
            
        Returns:
            시뮬레이션 결과 딕셔너리
        """
        # 1. 인기 상품 Top 10 추출
        popular_items = self.candidate_gen.generate_popularity_candidates(top_k=10)

        # 2. 평가 (모드에 따라)
        if self.mode == 'llm':
            # 메타데이터가 없으면 로드
            if user_metadata is None:
                con = self._connect_ro()
                user_metadata = load_user_metadata(con, user_id)
                con.close()
            
            clicked = self._evaluate_with_llm(user_id, popular_items, user_metadata)
        else:
            clicked = self._evaluate_fast(popular_items)
        
        # 3. 구매 시뮬레이션
        if clicked:
            # 구매 확률: 클릭의 3%가 구매로 전환
            purchase_count = 1 if random.random() < 0.03 else 0
        else:
            purchase_count = 0
        
        result = {
            'clicked': clicked,
            'items': popular_items,
            'num_items': len(popular_items),
            'mode': self.mode,
            'purchase_count': purchase_count
        }
        
        if self.mode == 'llm' and user_metadata:
            # dataclass to dict
            result['user_metadata'] = {
                'age_group': user_metadata.age_group,
                'purchase_frequency': user_metadata.purchase_frequency,
                'avg_price': user_metadata.avg_price,
                'recency': user_metadata.recency
            }
            
        return result
    
    def simulate_group_b(self, user_id: str, user_metadata: Optional[UserMetadata] = None) -> Dict[str, Any]:
        """
        Group B (Test) 시뮬레이션: ML 추천
        
        Args:
            user_id: 유저 ID
            user_metadata: 유저 메타데이터 (LLM 모드에서 사용, 옵션)
            
        Returns:
            시뮬레이션 결과 딕셔너리
        """
        # 1. ML 모델 추천 생성
        recommendations = self.rec_service.recommend(user_id, top_k=10)
        
        rec_items = recommendations.get('recommendations', [])
        
        # 2. 평가 (모드에 따라)
        if rec_items:
            if self.mode == 'llm':
                # 메타데이터가 없으면 로드 (simulation_group_a와 동일 패턴)
                if user_metadata is None:
                    con = self._connect_ro()
                    user_metadata = load_user_metadata(con, user_id)
                    con.close()

                clicked = self._evaluate_with_llm(user_id, rec_items, user_metadata)
            else:
                # FAST 모드: ML 추천은 인기 상품보다 20% 높은 클릭률
                base_clicked = self._evaluate_fast(rec_items)
                # ML 모델 보너스 (추가 10% 확률)
                ml_bonus = random.random() < 0.1
                clicked = base_clicked or ml_bonus
        else:
            clicked = False
        
        # 3. 구매 시뮬레이션
        if clicked:
            # 클릭한 경우: 구매 확률: 클릭의 40%가 구매로 전환 (개인화로 인해 Group A보다 높음)
            purchase_count = 1 if random.random() < 0.4 else 0
        else:
            # 클릭하지 않은 경우
            purchase_count = 0
        
        result = {
            'clicked': clicked,
            'items': rec_items,
            'num_items': len(rec_items),
            'mode': self.mode,
            'strategy': recommendations.get('strategy', 'unknown'),
            'segment': recommendations.get('segment', 'unknown'),
            'purchase_count': purchase_count
        }

        if self.mode == 'llm' and user_metadata:
            result['user_metadata'] = {
                'age_group': user_metadata.age_group,
                'purchase_frequency': user_metadata.purchase_frequency,
                'avg_price': user_metadata.avg_price,
                'recency': user_metadata.recency
            }
            
        return result
    
    def close(self):
        """리소스 정리"""
        if self.rec_service:
            self.rec_service.close()
        if self.candidate_gen:
            self.candidate_gen.close()


if __name__ == "__main__":
    # 테스트
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.simulation.ab_test import ABTestSimulator
    from src.models.serving import RecommendationService
    from src.models.candidate_generation import CandidateGenerator
    
    # FAST 모드 테스트
    logger.info("=" * 60)
    logger.info("FAST 모드 테스트")
    logger.info("=" * 60)
    
    simulator_fast = ABTestSimulator(
        ollama_client=None,
        rec_service=RecommendationService(),
        candidate_gen=CandidateGenerator(),
        mode='fast'
    )
    
    # 샘플 유저로 테스트
    con = duckdb.connect('local_helix.db')
    sample_user = con.execute("""
        SELECT customer_id 
        FROM user_features
        LIMIT 1
    """).fetchone()[0]
    con.close()
    
    # Group A 테스트
    logger.info("\n[Group A - FAST]")
    result_a = simulator_fast.simulate_group_a(sample_user)
    logger.info(f"Clicked: {result_a['clicked']}")
    logger.info(f"Items: {len(result_a['items'])}개")
    
    # Group B 테스트
    logger.info("\n[Group B - FAST]")
    result_b = simulator_fast.simulate_group_b(sample_user)
    logger.info(f"Clicked: {result_b['clicked']}")
    logger.info(f"Items: {len(result_b['items'])}개")
    
    logger.info("=" * 60)
    
    simulator_fast.close()

