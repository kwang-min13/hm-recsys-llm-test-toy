"""
Real Shopping Pattern Extraction from H&M Data

H&M 실제 데이터에서 쇼핑 패턴 추출:
1. 나이-구매 패턴
2. 성별-카테고리 선호도
3. 구매 빈도 분포
4. 가격대별 구매 패턴
"""

import sys
from pathlib import Path
import duckdb
import json
from collections import defaultdict, Counter

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("EXTRACTING REAL SHOPPING PATTERNS FROM H&M DATA")
print("=" * 80)

# DuckDB 연결
con = duckdb.connect(":memory:")
con.execute("SET memory_limit='8GB'")
con.execute("SET threads TO 4")

print("\n[1/6] Loading data...")
print("-" * 80)

# 데이터 로드
try:
    # Transactions
    con.execute("""
        CREATE VIEW transactions AS
        SELECT 
            customer_id::VARCHAR AS customer_id,
            article_id::VARCHAR AS article_id,
            CAST(t_dat AS DATE) AS t_dat,
            price::DOUBLE AS price
        FROM read_csv_auto('data/transactions_train.csv', header=true)
    """)
    
    # User features (from customers.csv for age data)
    con.execute("""
        CREATE VIEW users AS
        SELECT 
            customer_id::VARCHAR AS customer_id,
            age::INT AS age
        FROM read_csv_auto('data/customers.csv', header=true)
        WHERE age IS NOT NULL AND age BETWEEN 18 AND 100
    """)
    
    # Item features (from articles.csv for category data)
    con.execute("""
        CREATE VIEW items AS
        SELECT 
            article_id::VARCHAR AS article_id,
            product_type_name::VARCHAR AS category,
            CASE 
                WHEN product_type_name LIKE '%T-shirt%' OR product_type_name LIKE '%Top%' OR product_type_name LIKE '%Blouse%' THEN 'tops'
                WHEN product_type_name LIKE '%Trouser%' OR product_type_name LIKE '%Jeans%' OR product_type_name LIKE '%Shorts%' THEN 'bottoms'
                WHEN product_type_name LIKE '%Dress%' THEN 'dresses'
                WHEN product_type_name LIKE '%Shoe%' OR product_type_name LIKE '%Sneaker%' OR product_type_name LIKE '%Boot%' THEN 'shoes'
                WHEN product_type_name LIKE '%Bag%' OR product_type_name LIKE '%Belt%' OR product_type_name LIKE '%Hat%' OR product_type_name LIKE '%Jewellery%' THEN 'accessories'
                WHEN product_type_name LIKE '%Jacket%' OR product_type_name LIKE '%Coat%' OR product_type_name LIKE '%Cardigan%' THEN 'outerwear'
                ELSE 'other'
            END AS category_group
        FROM read_csv_auto('data/articles.csv', header=true)
    """)
    
    print("[OK] Data loaded successfully")
    
    # 데이터 크기 확인
    trans_count = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    user_count = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    item_count = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    
    print(f"  Transactions: {trans_count:,}")
    print(f"  Users: {user_count:,}")
    print(f"  Items: {item_count:,}")
    
except Exception as e:
    print(f"[ERROR] Error loading data: {e}")
    exit(1)

# Pattern 1: 나이-가격대 상관관계
print("\n[2/6] Analyzing Age-Price Correlation...")
print("-" * 80)

age_price_query = """
WITH user_purchases AS (
    SELECT 
        u.age,
        t.price,
        CASE 
            WHEN u.age <= 25 THEN '18-25'
            WHEN u.age <= 35 THEN '26-35'
            WHEN u.age <= 50 THEN '36-50'
            ELSE '51-65'
        END AS age_group,
        CASE 
            WHEN t.price < 0.02 THEN 'low'
            WHEN t.price < 0.04 THEN 'medium'
            ELSE 'high'
        END AS price_tier
    FROM transactions t
    JOIN users u ON t.customer_id = u.customer_id
    WHERE t.price IS NOT NULL AND t.price > 0
)
SELECT 
    age_group,
    price_tier,
    COUNT(*) AS count
FROM user_purchases
GROUP BY age_group, price_tier
ORDER BY age_group, price_tier
"""

age_price_results = con.execute(age_price_query).fetchall()

age_price_dist = defaultdict(lambda: defaultdict(int))
for age_group, price_tier, count in age_price_results:
    age_price_dist[age_group][price_tier] = count

print("\nAge-Price Distribution:")
for age_group in ['18-25', '26-35', '36-50', '51-65']:
    total = sum(age_price_dist[age_group].values())
    if total > 0:
        low_pct = age_price_dist[age_group]['low'] / total * 100
        med_pct = age_price_dist[age_group]['medium'] / total * 100
        high_pct = age_price_dist[age_group]['high'] / total * 100
        print(f"  {age_group}: low={low_pct:.1f}%, medium={med_pct:.1f}%, high={high_pct:.1f}%")

# Pattern 2: 나이-카테고리 선호도
print("\n[3/6] Analyzing Age-Category Preferences...")
print("-" * 80)

age_category_query = """
WITH user_purchases AS (
    SELECT 
        u.age,
        i.category_group,
        CASE 
            WHEN u.age <= 25 THEN '18-25'
            WHEN u.age <= 35 THEN '26-35'
            WHEN u.age <= 50 THEN '36-50'
            ELSE '51-65'
        END AS age_group
    FROM transactions t
    JOIN users u ON t.customer_id = u.customer_id
    JOIN items i ON t.article_id = i.article_id
    WHERE i.category_group != 'other'
)
SELECT 
    age_group,
    category_group,
    COUNT(*) AS count
FROM user_purchases
GROUP BY age_group, category_group
ORDER BY age_group, category_group
"""

age_category_results = con.execute(age_category_query).fetchall()

age_category_dist = defaultdict(lambda: defaultdict(int))
for age_group, category, count in age_category_results:
    age_category_dist[age_group][category] = count

print("\nAge-Category Distribution:")
for age_group in ['18-25', '26-35', '36-50', '51-65']:
    total = sum(age_category_dist[age_group].values())
    if total > 0:
        print(f"  {age_group}:")
        for cat in ['tops', 'bottoms', 'dresses', 'shoes', 'accessories', 'outerwear']:
            pct = age_category_dist[age_group][cat] / total * 100
            print(f"    {cat}: {pct:.1f}%")

# Pattern 3: 구매 빈도 분포
print("\n[4/6] Analyzing Purchase Frequency...")
print("-" * 80)

frequency_query = """
WITH user_purchase_counts AS (
    SELECT 
        customer_id,
        COUNT(*) AS purchase_count,
        DATE_DIFF('day', MIN(t_dat), MAX(t_dat)) AS days_active
    FROM transactions
    GROUP BY customer_id
    HAVING days_active > 0
),
user_frequency AS (
    SELECT 
        customer_id,
        purchase_count,
        days_active,
        purchase_count::DOUBLE / (days_active / 30.0) AS purchases_per_month,
        CASE 
            WHEN purchase_count::DOUBLE / (days_active / 30.0) >= 4 THEN 'weekly'
            WHEN purchase_count::DOUBLE / (days_active / 30.0) >= 1 THEN 'monthly'
            ELSE 'occasionally'
        END AS frequency
    FROM user_purchase_counts
)
SELECT 
    frequency,
    COUNT(*) AS user_count
FROM user_frequency
GROUP BY frequency
ORDER BY frequency
"""

frequency_results = con.execute(frequency_query).fetchall()

freq_dist = {freq: count for freq, count in frequency_results}
total_users = sum(freq_dist.values())

print("\nPurchase Frequency Distribution:")
for freq in ['weekly', 'monthly', 'occasionally']:
    pct = freq_dist.get(freq, 0) / total_users * 100
    print(f"  {freq}: {pct:.1f}%")

# Pattern 4: 가격대-구매빈도 상관관계
print("\n[5/6] Analyzing Price-Frequency Correlation...")
print("-" * 80)

price_freq_query = """
WITH user_avg_price AS (
    SELECT 
        customer_id,
        AVG(price) AS avg_price,
        CASE 
            WHEN AVG(price) < 0.02 THEN 'low'
            WHEN AVG(price) < 0.04 THEN 'medium'
            ELSE 'high'
        END AS price_tier
    FROM transactions
    WHERE price IS NOT NULL AND price > 0
    GROUP BY customer_id
),
user_frequency AS (
    SELECT 
        customer_id,
        COUNT(*) AS purchase_count,
        DATE_DIFF('day', MIN(t_dat), MAX(t_dat)) AS days_active
    FROM transactions
    GROUP BY customer_id
    HAVING days_active > 0
),
combined AS (
    SELECT 
        p.price_tier,
        CASE 
            WHEN f.purchase_count::DOUBLE / (f.days_active / 30.0) >= 4 THEN 'weekly'
            WHEN f.purchase_count::DOUBLE / (f.days_active / 30.0) >= 1 THEN 'monthly'
            ELSE 'occasionally'
        END AS frequency
    FROM user_avg_price p
    JOIN user_frequency f ON p.customer_id = f.customer_id
)
SELECT 
    price_tier,
    frequency,
    COUNT(*) AS count
FROM combined
GROUP BY price_tier, frequency
ORDER BY price_tier, frequency
"""

price_freq_results = con.execute(price_freq_query).fetchall()

price_freq_dist = defaultdict(lambda: defaultdict(int))
for price_tier, frequency, count in price_freq_results:
    price_freq_dist[price_tier][frequency] = count

print("\nPrice-Frequency Distribution:")
for price_tier in ['low', 'medium', 'high']:
    total = sum(price_freq_dist[price_tier].values())
    if total > 0:
        weekly_pct = price_freq_dist[price_tier]['weekly'] / total * 100
        monthly_pct = price_freq_dist[price_tier]['monthly'] / total * 100
        occ_pct = price_freq_dist[price_tier]['occasionally'] / total * 100
        print(f"  {price_tier}: weekly={weekly_pct:.1f}%, monthly={monthly_pct:.1f}%, occasionally={occ_pct:.1f}%")

# Pattern 5: 전체 카테고리 분포
print("\n[6/6] Analyzing Overall Category Distribution...")
print("-" * 80)

category_query = """
SELECT 
    category_group,
    COUNT(*) AS count
FROM transactions t
JOIN items i ON t.article_id = i.article_id
WHERE category_group != 'other'
GROUP BY category_group
ORDER BY count DESC
"""

category_results = con.execute(category_query).fetchall()

total_purchases = sum(count for _, count in category_results)

print("\nOverall Category Distribution:")
for category, count in category_results:
    pct = count / total_purchases * 100
    print(f"  {category}: {pct:.1f}%")

# 결과를 JSON으로 저장
print("\n[Saving Results]")
print("-" * 80)

patterns = {
    "age_price_distribution": {
        age_group: {
            "low": age_price_dist[age_group]['low'] / sum(age_price_dist[age_group].values()) * 100,
            "medium": age_price_dist[age_group]['medium'] / sum(age_price_dist[age_group].values()) * 100,
            "high": age_price_dist[age_group]['high'] / sum(age_price_dist[age_group].values()) * 100
        }
        for age_group in ['18-25', '26-35', '36-50', '51-65']
        if sum(age_price_dist[age_group].values()) > 0
    },
    "age_category_distribution": {
        age_group: {
            cat: age_category_dist[age_group][cat] / sum(age_category_dist[age_group].values()) * 100
            for cat in ['tops', 'bottoms', 'dresses', 'shoes', 'accessories', 'outerwear']
        }
        for age_group in ['18-25', '26-35', '36-50', '51-65']
        if sum(age_category_dist[age_group].values()) > 0
    },
    "price_frequency_distribution": {
        price_tier: {
            "weekly": price_freq_dist[price_tier]['weekly'] / sum(price_freq_dist[price_tier].values()) * 100,
            "monthly": price_freq_dist[price_tier]['monthly'] / sum(price_freq_dist[price_tier].values()) * 100,
            "occasionally": price_freq_dist[price_tier]['occasionally'] / sum(price_freq_dist[price_tier].values()) * 100
        }
        for price_tier in ['low', 'medium', 'high']
        if sum(price_freq_dist[price_tier].values()) > 0
    },
    "overall_frequency": {
        freq: freq_dist.get(freq, 0) / total_users * 100
        for freq in ['weekly', 'monthly', 'occasionally']
    },
    "overall_category": {
        category: count / total_purchases * 100
        for category, count in category_results
    }
}

output_path = Path('data/shopping_patterns.json')
with open(output_path, 'w') as f:
    json.dump(patterns, f, indent=2)

print("[OK] Patterns saved to: {}".format(output_path))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\n[OK] Real shopping patterns extracted from H&M data!")
print("[OK] Patterns saved to data/shopping_patterns.json")
print("\nNext steps:")
print("  1. Review extracted patterns")
print("  2. Update virtual_user.py with real distributions")
print("  3. Re-run A/B test with data-driven personas")
print("=" * 80)

con.close()
