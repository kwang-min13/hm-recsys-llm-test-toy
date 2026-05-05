"""
A/B Test Sample Size Calculator

A/B 테스트를 위한 적정 표본 크기 계산기

통계적 검정력 분석을 통해 필요한 샘플 크기를 계산합니다.
"""

import math
from typing import Dict, Any
from scipy import stats
import numpy as np


def calculate_sample_size_proportion(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True
) -> Dict[str, Any]:
    """
    비율(CTR, Conversion Rate 등)에 대한 표본 크기 계산
    
    Args:
        baseline_rate: 기준 그룹(Control)의 예상 비율 (0~1)
        minimum_detectable_effect: 감지하고자 하는 최소 효과 크기 (절대값, 0~1)
                                   예: 0.05 = 5%p 차이
        alpha: 유의수준 (Type I error rate, 기본값 0.05)
        power: 검정력 (1 - Type II error rate, 기본값 0.80)
        two_tailed: 양측 검정 여부 (기본값 True)
    
    Returns:
        Dict containing:
        - sample_size_per_group: 그룹당 필요한 샘플 크기
        - total_sample_size: 총 필요한 샘플 크기 (두 그룹 합)
        - baseline_rate: 입력된 기준 비율
        - expected_test_rate: 예상되는 테스트 그룹 비율
        - minimum_detectable_effect: 최소 감지 효과
        - alpha: 유의수준
        - power: 검정력
    """
    # Z-scores
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # 예상되는 테스트 그룹 비율
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    
    # 평균 비율
    p_avg = (p1 + p2) / 2
    
    # 표본 크기 계산 (각 그룹당)
    numerator = (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) + 
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2
    
    n_per_group = math.ceil(numerator / denominator)
    
    return {
        "sample_size_per_group": n_per_group,
        "total_sample_size": n_per_group * 2,
        "baseline_rate": baseline_rate,
        "expected_test_rate": p2,
        "minimum_detectable_effect": minimum_detectable_effect,
        "relative_lift": (minimum_detectable_effect / baseline_rate * 100) if baseline_rate > 0 else 0,
        "alpha": alpha,
        "power": power,
        "two_tailed": two_tailed
    }


def calculate_sample_size_mean(
    baseline_mean: float,
    baseline_std: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True
) -> Dict[str, Any]:
    """
    평균값(만족도, 구매액 등)에 대한 표본 크기 계산
    
    Args:
        baseline_mean: 기준 그룹의 예상 평균
        baseline_std: 기준 그룹의 예상 표준편차
        minimum_detectable_effect: 감지하고자 하는 최소 효과 크기 (절대값)
        alpha: 유의수준
        power: 검정력
        two_tailed: 양측 검정 여부
    
    Returns:
        Dict containing sample size information
    """
    # Z-scores
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # 표본 크기 계산 (각 그룹당)
    # 두 그룹의 표준편차가 같다고 가정
    pooled_std = baseline_std
    
    n_per_group = math.ceil(
        2 * ((z_alpha + z_beta) * pooled_std / minimum_detectable_effect) ** 2
    )
    
    return {
        "sample_size_per_group": n_per_group,
        "total_sample_size": n_per_group * 2,
        "baseline_mean": baseline_mean,
        "expected_test_mean": baseline_mean + minimum_detectable_effect,
        "baseline_std": baseline_std,
        "minimum_detectable_effect": minimum_detectable_effect,
        "effect_size_cohen_d": minimum_detectable_effect / baseline_std if baseline_std > 0 else 0,
        "alpha": alpha,
        "power": power,
        "two_tailed": two_tailed
    }


def calculate_sample_size_for_ab_test(
    baseline_ctr: float = 0.75,
    target_lift: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.80
) -> Dict[str, Any]:
    """
    A/B 테스트 시나리오에 맞춘 표본 크기 계산
    
    Args:
        baseline_ctr: Control 그룹의 예상 CTR (0~1)
        target_lift: 목표 상승률 (상대적, 예: 0.05 = 5% 상승)
        alpha: 유의수준
        power: 검정력
    
    Returns:
        Dict containing comprehensive sample size analysis
    """
    # 절대 효과 크기 계산
    absolute_effect = baseline_ctr * target_lift
    
    # CTR에 대한 표본 크기
    ctr_result = calculate_sample_size_proportion(
        baseline_rate=baseline_ctr,
        minimum_detectable_effect=absolute_effect,
        alpha=alpha,
        power=power
    )
    
    # 구매 전환율에 대한 표본 크기 (예시)
    # 가정: CTR의 50%가 구매로 전환
    baseline_conversion = baseline_ctr * 0.5
    conversion_effect = baseline_conversion * target_lift
    
    conversion_result = calculate_sample_size_proportion(
        baseline_rate=baseline_conversion,
        minimum_detectable_effect=conversion_effect,
        alpha=alpha,
        power=power
    )
    
    # 최대값 선택 (가장 보수적인 추정)
    max_sample_size = max(
        ctr_result['sample_size_per_group'],
        conversion_result['sample_size_per_group'],
    )
    
    return {
        "recommended_sample_size_per_group": max_sample_size,
        "recommended_total_sample_size": max_sample_size * 2,
        "ctr_analysis": ctr_result,
        "conversion_analysis": conversion_result,
        "parameters": {
            "baseline_ctr": baseline_ctr,
            "target_lift": target_lift,
            "alpha": alpha,
            "power": power
        }
    }


def print_sample_size_report(result: Dict[str, Any]):
    """표본 크기 계산 결과를 보기 좋게 출력"""
    print("=" * 80)
    print("A/B TEST SAMPLE SIZE CALCULATION REPORT")
    print("=" * 80)
    
    params = result['parameters']
    print(f"\nInput Parameters:")
    print(f"  Baseline CTR: {params['baseline_ctr']:.1%}")
    print(f"  Target Lift: {params['target_lift']:.1%}")
    print(f"  Significance Level (alpha): {params['alpha']:.2f}")
    print(f"  Statistical Power: {params['power']:.1%}")
    
    print(f"\n{'='*80}")
    print("RECOMMENDED SAMPLE SIZE")
    print("=" * 80)
    print(f"  Per Group: {result['recommended_sample_size_per_group']:,} users")
    print(f"  Total: {result['recommended_total_sample_size']:,} users")
    
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS BY METRIC")
    print("=" * 80)
    
    # CTR Analysis
    ctr = result['ctr_analysis']
    print(f"\n1. Click-Through Rate (CTR)")
    print(f"   Baseline: {ctr['baseline_rate']:.1%}")
    print(f"   Expected Test: {ctr['expected_test_rate']:.1%}")
    print(f"   Absolute Difference: {ctr['minimum_detectable_effect']:.1%}")
    print(f"   Relative Lift: {ctr['relative_lift']:.1f}%")
    print(f"   Required Sample Size: {ctr['sample_size_per_group']:,} per group")
    
    # Conversion Analysis
    conv = result['conversion_analysis']
    print(f"\n2. Purchase Conversion Rate")
    print(f"   Baseline: {conv['baseline_rate']:.1%}")
    print(f"   Expected Test: {conv['expected_test_rate']:.1%}")
    print(f"   Absolute Difference: {conv['minimum_detectable_effect']:.1%}")
    print(f"   Relative Lift: {conv['relative_lift']:.1f}%")
    print(f"   Required Sample Size: {conv['sample_size_per_group']:,} per group")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print("=" * 80)
    print(f"\nTo detect a {params['target_lift']:.1%} relative improvement with:")
    print(f"  - {params['power']:.0%} probability (statistical power)")
    print(f"  - {params['alpha']:.0%} significance level")
    print(f"\nYou need at least {result['recommended_sample_size_per_group']:,} users per group")
    print(f"(Total: {result['recommended_total_sample_size']:,} users)")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    # 예시 1: 기본 시나리오
    print("\n[Example 1] Default Scenario")
    print("-" * 80)
    result1 = calculate_sample_size_for_ab_test(
        baseline_ctr=0.75,  # 75% CTR
        target_lift=0.05,   # 5% 상승 목표
        alpha=0.05,
        power=0.80
    )
    print_sample_size_report(result1)
    
    # 예시 2: 더 작은 효과 감지
    print("\n\n[Example 2] Smaller Effect (2% lift)")
    print("-" * 80)
    result2 = calculate_sample_size_for_ab_test(
        baseline_ctr=0.75,
        target_lift=0.02,   # 2% 상승 목표 (더 작은 효과)
        alpha=0.05,
        power=0.80
    )
    print_sample_size_report(result2)
    
    # 예시 3: 더 높은 검정력
    print("\n\n[Example 3] Higher Power (90%)")
    print("-" * 80)
    result3 = calculate_sample_size_for_ab_test(
        baseline_ctr=0.75,
        target_lift=0.05,
        alpha=0.05,
        power=0.90  # 90% 검정력
    )
    print_sample_size_report(result3)
    
    # 예시 4: 현재 A/B 테스트 결과 기반
    print("\n\n[Example 4] Based on Current A/B Test Results")
    print("-" * 80)
    print("\nCurrent results: Group A CTR=78.08%, Group B CTR=74.03%")
    print("Observed difference: -4.05%p (Group B underperformed)")
    print("\nTo detect this difference as statistically significant:")
    
    result4 = calculate_sample_size_proportion(
        baseline_rate=0.7808,
        minimum_detectable_effect=0.0405,  # 4.05%p 차이
        alpha=0.05,
        power=0.80
    )
    
    print(f"  Required sample size: {result4['sample_size_per_group']:,} per group")
    print(f"  Current sample size: 511 (Group A), 489 (Group B)")
    print(f"  Conclusion: Current sample (1,000 total) is {'SUFFICIENT' if result4['total_sample_size'] <= 1000 else 'INSUFFICIENT'}")
    
    print("\n" + "=" * 80)
