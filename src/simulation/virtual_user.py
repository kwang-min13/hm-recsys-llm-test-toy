"""
Virtual User Module

LLM 기반 가상 유저 페르소나 생성 (optimized)
"""

import random
from typing import Dict, Any, List, Optional
import logging
import json
import re

from .ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VirtualUser:
    """가상 유저 클래스"""

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        """
        Args:
            ollama_client: Ollama 클라이언트 (선택사항)
              - None이면 LLM을 사용하지 않음(중요: 기존 코드는 None이어도 OllamaClient를 생성했음)
        """
        self.ollama_client = ollama_client  # ✅ None이면 LLM 완전 미사용
        self._llm_available: Optional[bool] = None  # ✅ 연결 체크 캐시
        self.persona: Dict[str, Any] = {}

    def _is_llm_available(self) -> bool:
        """Ollama 연결 가능 여부를 1회만 확인(캐시)"""
        if self.ollama_client is None:
            self._llm_available = False
            return False
        if self._llm_available is None:
            try:
                self._llm_available = bool(self.ollama_client.check_connection())
            except Exception:
                self._llm_available = False
        return self._llm_available

    def generate_persona(self) -> Dict[str, Any]:
        """가상 유저 페르소나 생성"""
        age = random.randint(18, 65)
        gender = random.choice(["Male", "Female", "Non-binary"])

        if self._is_llm_available():
            # ✅ JSON만 출력 강제 (장문 설명 방지)
            prompt = (
                f"Generate a realistic shopping persona for a {age}-year-old {gender} customer.\n"
                "Return ONLY valid JSON. No prose, no markdown, no code fences.\n"
                "Keys: style, frequency, budget, categories.\n"
                "Constraints:\n"
                "- style: casual|formal|sporty|trendy|vintage\n"
                "- frequency: weekly|monthly|occasionally\n"
                "- budget: low|medium|high\n"
                "- categories: array of 2-3 strings\n"
            )

            # ✅ 핵심 최적화: 출력 길이 제한 + 불필요한 추가 문단 차단
            response = self.ollama_client.generate(
                prompt,
                temperature=0.6,
                num_predict=140,
                stop=["\n\n"]
            )
            persona_details = self._parse_persona_json(response) if response else self._fallback_persona()
        else:
            persona_details = self._fallback_persona()

        self.persona = {"age": age, "gender": gender, **persona_details}
        return self.persona

    def _parse_persona_json(self, response: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON만 안전 추출 + 최소 스키마 보정"""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1 or end <= start:
                return self._fallback_persona()

            obj = json.loads(response[start:end])

            styles = {"casual", "formal", "sporty", "trendy", "vintage"}
            freqs = {"weekly", "monthly", "occasionally"}
            budgets = {"low", "medium", "high"}

            style = obj.get("style", "casual")
            frequency = obj.get("frequency", "occasionally")
            budget = obj.get("budget", "medium")
            categories = obj.get("categories", ["tops", "bottoms"])

            if style not in styles:
                style = "casual"
            if frequency not in freqs:
                frequency = "occasionally"
            if budget not in budgets:
                budget = "medium"

            if not isinstance(categories, list) or len(categories) < 1:
                categories = ["tops", "bottoms"]
            categories = [str(x) for x in categories][:3]

            return {
                "style": style,
                "frequency": frequency,
                "budget": budget,
                "categories": categories,
            }
        except Exception:
            return self._fallback_persona()

    def _fallback_persona(self) -> Dict[str, Any]:
        """
        H&M 실제 데이터 기반 페르소나 생성
        
        데이터 소스: 31.7M 거래, 1.3M 사용자
        반영된 패턴:
        1. 나이-예산 분포 (약한 상관관계)
        2. 전체 카테고리 분포 (tops/bottoms 압도적)
        3. 전체 구매 빈도 분포 (monthly 53%)
        4. 예산-빈도 상관관계 (약한 상관관계)
        """
        # 기본 속성 생성 (이미 generate_persona에서 생성됨)
        # 여기서는 age를 참조하기 위해 self.persona에서 가져옴
        age = self.persona.get('age', 30) if self.persona else 30
        gender = self.persona.get('gender', 'Male') if self.persona else 'Male'
        
        # 1. 나이 기반 예산 분포 (실제 H&M 데이터)
        budget = self._get_budget_by_age_real(age)
        
        # 2. 스타일 선택 (데이터 없음 - 균등 분포 사용)
        styles = ["casual", "formal", "sporty", "trendy", "vintage"]
        style = random.choice(styles)
        
        # 3. 예산 기반 구매 빈도 (실제 H&M 데이터)
        frequency = self._get_frequency_by_budget_real(budget)
        
        # 4. 전체 카테고리 분포 (실제 H&M 데이터)
        categories = self._get_categories_real()

        return {
            "style": style,
            "frequency": frequency,
            "budget": budget,
            "categories": categories,
        }

    def _get_budget_by_age_real(self, age: int) -> str:
        """
        실제 H&M 데이터 기반 나이-예산 분포
        
        데이터 출처: data/shopping_patterns.json
        - 18-25: low=41.2%, medium=42.1%, high=16.6%
        - 26-35: low=38.7%, medium=41.0%, high=20.4%
        - 36-50: low=39.8%, medium=41.2%, high=19.0%
        - 51-65: low=35.2%, medium=42.8%, high=22.0%
        """
        if age <= 25:
            # 18-25세: low 약간 높음
            return random.choices(
                ["low", "medium", "high"],
                weights=[41.2, 42.1, 16.6]
            )[0]
        elif age <= 35:
            # 26-35세: 균형잡힌 분포
            return random.choices(
                ["low", "medium", "high"],
                weights=[38.7, 41.0, 20.4]
            )[0]
        elif age <= 50:
            # 36-50세: medium 우세
            return random.choices(
                ["low", "medium", "high"],
                weights=[39.8, 41.2, 19.0]
            )[0]
        else:  # 51-65
            # 51-65세: high 약간 증가
            return random.choices(
                ["low", "medium", "high"],
                weights=[35.2, 42.8, 22.0]
            )[0]

    def _get_frequency_by_budget_real(self, budget: str) -> str:
        """
        실제 H&M 데이터 기반 예산-빈도 분포
        
        데이터 출처: data/shopping_patterns.json
        - low: weekly=22.0%, monthly=51.8%, occasionally=26.2%
        - medium: weekly=21.7%, monthly=55.1%, occasionally=23.2%
        - high: weekly=24.6%, monthly=40.8%, occasionally=34.6%
        """
        if budget == "low":
            return random.choices(
                ["weekly", "monthly", "occasionally"],
                weights=[22.0, 51.8, 26.2]
            )[0]
        elif budget == "medium":
            return random.choices(
                ["weekly", "monthly", "occasionally"],
                weights=[21.7, 55.1, 23.2]
            )[0]
        else:  # high
            return random.choices(
                ["weekly", "monthly", "occasionally"],
                weights=[24.6, 40.8, 34.6]
            )[0]

    def _get_categories_real(self) -> List[str]:
        """
        실제 H&M 데이터 기반 전체 카테고리 분포
        
        데이터 출처: data/shopping_patterns.json
        - tops: 33.4%
        - bottoms: 33.9%
        - dresses: 20.4%
        - shoes: 2.0%
        - accessories: 3.4%
        - outerwear: 6.9%
        
        참고: 나이/성별과 무관하게 유사한 분포
        """
        all_categories = ["tops", "bottoms", "dresses", "shoes", "accessories", "outerwear"]
        
        # 실제 H&M 전체 분포 (반올림)
        weights = [33.4, 33.9, 20.4, 2.0, 3.4, 6.9]
        
        # 가중치 기반으로 2개 선택
        selected = []
        while len(selected) < 2:
            cat = random.choices(all_categories, weights=weights)[0]
            if cat not in selected:
                selected.append(cat)
        
        return selected

    def evaluate_recommendations(self, recommendations: List[str]) -> Dict[str, Any]:
        """추천 상품 평가"""
        if not self.persona:
            self.generate_persona()

        n = len(recommendations)
        if n == 0:
            return {"purchase_count": 0, "satisfaction": 3, "acceptance_rate": 0.0}

        purchase_count = 0
        satisfaction = 3

        if self._is_llm_available():
            # ✅ 한 줄만 출력 강제 (장문 응답 방지)
            prompt = (
                f"You are a {self.persona['age']}-year-old {self.persona['gender']} shopper.\n"
                f"Style: {self.persona.get('style','casual')}. "
                f"Budget: {self.persona.get('budget','medium')}. "
                f"Favorite categories: {', '.join(self.persona.get('categories',['general']))}.\n"
                f"You received {n} product recommendations.\n"
                f"Return ONLY: Purchase: X, Satisfaction: Y (X is 0-{n}, Y is 1-5)."
            )

            response = self.ollama_client.generate(
                prompt,
                temperature=0.3,
                num_predict=50,
                stop=["\n"]
            )

            purchase_count, satisfaction = self._parse_eval(response, n)
        else:
            purchase_count, satisfaction = self._random_eval(n)

        return {
            "purchase_count": purchase_count,
            "satisfaction": satisfaction,
            "acceptance_rate": purchase_count / n if n else 0.0,
        }

    def _parse_eval(self, response: Optional[str], n: int):
        """Purchase/Satisfaction 파싱"""
        if response:
            try:
                pm = re.search(r"Purchase:\s*(\d+)", response)
                sm = re.search(r"Satisfaction:\s*(\d+)", response)
                purchase_count = min(int(pm.group(1)), n) if pm else 0
                satisfaction = max(1, min(int(sm.group(1)), 5)) if sm else 3
                return purchase_count, satisfaction
            except Exception:
                pass
        return self._random_eval(n)

    def _random_eval(self, n: int):
        purchase_count = random.randint(0, min(3, n))
        satisfaction = random.randint(2, 5)
        return purchase_count, satisfaction
