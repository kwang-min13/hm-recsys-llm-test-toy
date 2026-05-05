"""
Statistical Analysis Module

A/B 테스트 결과에 대한 통계 분석 수행
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
from typing import Dict, Any


def load_ab_test_results(filepath: str = 'logs/ab_test_results_llm.csv') -> pd.DataFrame:
    """
    A/B 테스트 결과 로드
    
    Args:
        filepath: 결과 CSV 파일 경로
        
    Returns:
        결과 데이터프레임
    """
    df = pd.read_csv(filepath)
    return df


def calculate_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    기본 통계 계산
    
    Args:
        df: A/B 테스트 결과 데이터프레임
        
    Returns:
        기본 통계 딕셔너리
    """
    group_a = df[df['group'] == 'A']
    group_b = df[df['group'] == 'B']
    
    stats = {
        'total_users': len(df),
        'group_a_size': len(group_a),
        'group_b_size': len(group_b),
        'group_a_ctr': group_a['clicked'].mean(),
        'group_b_ctr': group_b['clicked'].mean(),
        'ctr_difference': group_b['clicked'].mean() - group_a['clicked'].mean(),
        'ctr_lift': ((group_b['clicked'].mean() - group_a['clicked'].mean()) / group_a['clicked'].mean()) * 100 if group_a['clicked'].mean() > 0 else 0,
        'group_a_avg_purchases': group_a['purchase_count'].mean(),
        'group_b_avg_purchases': group_b['purchase_count'].mean(),
    }
    
    return stats


def chi_square_test(df: pd.DataFrame) -> Dict[str, Any]:
    """
    카이제곱 검정으로 A/B 그룹 간 CTR 차이 검증
    
    Args:
        df: A/B 테스트 결과 데이터프레임
        
    Returns:
        검정 결과 딕셔너리
    """
    # 분할표 생성
    contingency_table = pd.crosstab(
        df['group'],
        df['clicked']
    )
    
    # 카이제곱 검정
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    return {
        'test_name': 'Chi-Square Test',
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'significant': p_value < 0.05,
        'significance_level': 0.05,
        'contingency_table': contingency_table.to_dict()
    }

def t_test_purchases(df: pd.DataFrame) -> Dict[str, Any]:
    """
    T-검정으로 구매 수 차이 검증
    
    Args:
        df: A/B 테스트 결과 데이터프레임
        
    Returns:
        검정 결과 딕셔너리
    """
    group_a_purchases = df[df['group'] == 'A']['purchase_count']
    group_b_purchases = df[df['group'] == 'B']['purchase_count']
    
    t_stat, p_value = ttest_ind(group_a_purchases, group_b_purchases)
    
    return {
        'test_name': 'Independent T-Test (Purchases)',
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'group_a_mean': group_a_purchases.mean(),
        'group_b_mean': group_b_purchases.mean(),
        'difference': group_b_purchases.mean() - group_a_purchases.mean()
    }


def analyze_ab_test(filepath: str = 'logs/ab_test_results_llm.csv') -> Dict[str, Any]:
    """
    전체 A/B 테스트 분석 수행
    
    Args:
        filepath: 결과 CSV 파일 경로
        
    Returns:
        전체 분석 결과 딕셔너리
    """
    # 데이터 로드
    df = load_ab_test_results(filepath)
    
    # 분석 수행
    basic_stats = calculate_basic_stats(df)
    chi2_result = chi_square_test(df)
    t_test_pur = t_test_purchases(df)
    
    return {
        'basic_stats': basic_stats,
        'chi_square_test': chi2_result,
        't_test_purchases': t_test_pur,
        'data_summary': {
            'total_records': len(df),
            'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        }
    }


def print_analysis_summary(results: Dict[str, Any]) -> None:
    """
    분석 결과 요약 출력
    
    Args:
        results: analyze_ab_test 결과
    """
    print("=" * 70)
    print("A/B Test Analysis Summary")
    print("=" * 70)
    
    stats = results['basic_stats']
    print(f"\n[*] Basic Statistics:")
    print(f"  Total Users: {stats['total_users']}")
    print(f"  Group A: {stats['group_a_size']} users")
    print(f"  Group B: {stats['group_b_size']} users")
    print(f"\n[*] Click-Through Rate (CTR):")
    print(f"  Group A CTR: {stats['group_a_ctr']:.2%}")
    print(f"  Group B CTR: {stats['group_b_ctr']:.2%}")
    print(f"  CTR Difference: {stats['ctr_difference']:.2%}")
    print(f"  CTR Lift: {stats['ctr_lift']:.2f}%")
    
    print(f"\n[*] Average Purchases:")
    print(f"  Group A: {stats['group_a_avg_purchases']:.2f}")
    print(f"  Group B: {stats['group_b_avg_purchases']:.2f}")
    
    chi2 = results['chi_square_test']
    print(f"\n[*] Chi-Square Test (CTR):")
    print(f"  Chi-square statistic: {chi2['chi2_statistic']:.4f}")
    print(f"  p-value: {chi2['p_value']:.4f}")
    print(f"  Significant: {'[YES]' if chi2['significant'] else '[NO]'} (alpha=0.05)")
    
    t_pur = results['t_test_purchases']
    print(f"\n[*] T-Test (Purchases):")
    print(f"  t-statistic: {t_pur['t_statistic']:.4f}")
    print(f"  p-value: {t_pur['p_value']:.4f}")
    print(f"  Significant: {'[YES]' if t_pur['significant'] else '[NO]'} (alpha=0.05)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # 분석 실행
    results = analyze_ab_test()
    print_analysis_summary(results)
