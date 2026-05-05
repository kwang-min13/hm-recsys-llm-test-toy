"""
Enhanced Virtual User Module with Realistic Shopping Patterns

개선된 Fallback 페르소나:
- 나이-예산 상관관계
- 나이-스타일 선호도
- 성별-카테고리 선호도
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
    """가상 유저 클래스 (Enhanced with realistic patterns)"""

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        """
        Args:
            ollama_client: Ollama 클라이언트 (선택사항)
              - None이면 LLM을 사용하지 않음
        """
        self.ollama_client = ollama_client
        self._llm_available: Optional[bool] = None
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
            # LLM 모드
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

            response = self.ollama_client.generate(
                prompt,
                temperature=0.6,
                num_predict=140,
                stop=["\n\n"]
            )
            persona_details = self._parse_persona_json(response) if response else self._fallback_persona_realistic(age, gender)
        else:
            # Enhanced Fallback 모드 (현실적 패턴 반영)
            persona_details = self._fallback_persona_realistic(age, gender)

        self.persona = {"age": age, "gender": gender, **persona_details}
        return self.persona

    def _parse_persona_json(self, response: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON만 안전 추출 + 최소 스키마 보정"""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1 or end <= start:
                return self._fallback_persona_realistic(30, "Male")  # Default fallback

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
            return self._fallback_persona_realistic(30, "Male")

    def _fallback_persona_realistic(self, age: int, gender: str) -> Dict[str, Any]:
        """
        현실적인 쇼핑 패턴을 반영한 Fallback 페르소나
        
        반영된 패턴:
        1. 나이-예산 상관관계 (젊을수록 low, 중년 medium, 장년 high)
        2. 나이-스타일 선호도 (젊을수록 trendy, 나이들수록 classic)
        3. 성별-카테고리 선호도
        4. 예산-구매빈도 상관관계
        """
        
        # 1. 나이 기반 예산 분포 (현실적 가중치)
        budget = self._get_budget_by_age(age)
        
        # 2. 나이 기반 스타일 선호도
        style = self._get_style_by_age(age)
        
        # 3. 예산 기반 구매 빈도
        frequency = self._get_frequency_by_budget(budget)
        
        # 4. 성별 기반 카테고리 선호도
        categories = self._get_categories_by_gender(gender)

        return {
            "style": style,
            "frequency": frequency,
            "budget": budget,
            "categories": categories,
        }

    def _get_budget_by_age(self, age: int) -> str:
        """
        나이에 따른 예산 분포 (현실적 가중치)
        
        - 18-25: low(60%), medium(30%), high(10%)
        - 26-35: low(30%), medium(50%), high(20%)
        - 36-50: low(20%), medium(40%), high(40%)
        - 51-65: low(15%), medium(35%), high(50%)
        """
        if age <= 25:
            return random.choices(
                ["low", "medium", "high"],
                weights=[60, 30, 10]
            )[0]
        elif age <= 35:
            return random.choices(
                ["low", "medium", "high"],
                weights=[30, 50, 20]
            )[0]
        elif age <= 50:
            return random.choices(
                ["low", "medium", "high"],
                weights=[20, 40, 40]
            )[0]
        else:  # 51-65
            return random.choices(
                ["low", "medium", "high"],
                weights=[15, 35, 50]
            )[0]

    def _get_style_by_age(self, age: int) -> str:
        """
        나이에 따른 스타일 선호도
        
        - 18-25: trendy(40%), casual(30%), sporty(20%), formal(5%), vintage(5%)
        - 26-35: casual(35%), trendy(25%), sporty(20%), formal(15%), vintage(5%)
        - 36-50: casual(40%), formal(25%), sporty(15%), trendy(10%), vintage(10%)
        - 51-65: casual(35%), formal(30%), vintage(20%), sporty(10%), trendy(5%)
        """
        if age <= 25:
            return random.choices(
                ["trendy", "casual", "sporty", "formal", "vintage"],
                weights=[40, 30, 20, 5, 5]
            )[0]
        elif age <= 35:
            return random.choices(
                ["casual", "trendy", "sporty", "formal", "vintage"],
                weights=[35, 25, 20, 15, 5]
            )[0]
        elif age <= 50:
            return random.choices(
                ["casual", "formal", "sporty", "trendy", "vintage"],
                weights=[40, 25, 15, 10, 10]
            )[0]
        else:  # 51-65
            return random.choices(
                ["casual", "formal", "vintage", "sporty", "trendy"],
                weights=[35, 30, 20, 10, 5]
            )[0]

    def _get_frequency_by_budget(self, budget: str) -> str:
        """
        예산에 따른 구매 빈도
        
        - low: occasionally(70%), monthly(25%), weekly(5%)
        - medium: monthly(50%), occasionally(30%), weekly(20%)
        - high: weekly(50%), monthly(35%), occasionally(15%)
        """
        if budget == "low":
            return random.choices(
                ["occasionally", "monthly", "weekly"],
                weights=[70, 25, 5]
            )[0]
        elif budget == "medium":
            return random.choices(
                ["monthly", "occasionally", "weekly"],
                weights=[50, 30, 20]
            )[0]
        else:  # high
            return random.choices(
                ["weekly", "monthly", "occasionally"],
                weights=[50, 35, 15]
            )[0]

    def _get_categories_by_gender(self, gender: str) -> List[str]:
        """
        성별에 따른 카테고리 선호도
        
        - Male: tops(30%), bottoms(25%), shoes(20%), outerwear(15%), accessories(10%)
        - Female: dresses(25%), tops(20%), shoes(20%), accessories(20%), bottoms(15%)
        - Non-binary: 균등 분포
        """
        all_categories = ["tops", "bottoms", "dresses", "shoes", "accessories", "outerwear"]
        
        if gender == "Male":
            # Male은 dresses 선호도 낮음
            weights = [30, 25, 5, 20, 10, 15]  # tops, bottoms, dresses, shoes, accessories, outerwear
        elif gender == "Female":
            # Female은 dresses, accessories 선호도 높음
            weights = [20, 15, 25, 20, 20, 10]
        else:  # Non-binary
            # 균등 분포
            weights = [20, 20, 15, 20, 15, 10]
        
        # 가중치 기반으로 2개 선택 (중복 없음)
        selected = random.choices(
            all_categories,
            weights=weights,
            k=2
        )
        
        # 중복 제거 (혹시 모를 경우)
        if selected[0] == selected[1]:
            # 다른 카테고리 선택
            remaining = [c for c in all_categories if c != selected[0]]
            selected[1] = random.choice(remaining)
        
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
