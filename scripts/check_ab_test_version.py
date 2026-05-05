import os
import time
from datetime import datetime

print('=' * 80)
print('A/B TEST vs MODEL TRAINING - TIMELINE ANALYSIS')
print('=' * 80)

files = {
    'A/B Test Results': 'logs/ab_test_results.csv',
    'Trained Model': 'models/artifacts/purchase_ranker.pkl',
    'Source: dataset.py': 'src/models/dataset.py',
    'Source: ranker.py': 'src/models/ranker.py',
}

print('\nFile Modification Times:')
print('-' * 80)

times = {}
for name, path in files.items():
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        times[name] = mtime
        dt = datetime.fromtimestamp(mtime)
        print(f'{name:25s}: {dt.strftime("%Y-%m-%d %H:%M:%S")}')
    else:
        print(f'{name:25s}: NOT FOUND')

print('\n' + '=' * 80)
print('ANALYSIS')
print('=' * 80)

if 'A/B Test Results' in times and 'Trained Model' in times:
    ab_time = times['A/B Test Results']
    model_time = times['Trained Model']
    
    time_diff = model_time - ab_time
    hours_diff = time_diff / 3600
    
    print(f'\nA/B Test timestamp: {datetime.fromtimestamp(ab_time)}')
    print(f'Model trained at:   {datetime.fromtimestamp(model_time)}')
    print(f'\nTime difference: {abs(hours_diff):.2f} hours')
    
    if ab_time > model_time:
        print(f'\n[NEWER] A/B test ran AFTER model training')
        print(f'        -> A/B test results reflect the NEW model [YES]')
    else:
        print(f'\n[OLDER] A/B test ran BEFORE model training')
        print(f'        -> A/B test results use OLD model [NO]')
        print(f'        -> Need to re-run A/B test with new model!')

# Check A/B test data timestamp
print('\n' + '=' * 80)
print('A/B TEST DATA INSPECTION')
print('=' * 80)

import pandas as pd
df = pd.read_csv('logs/ab_test_results.csv')
print(f'\nTotal records: {len(df)}')
print(f'First timestamp: {df["timestamp"].iloc[0]}')
print(f'Last timestamp:  {df["timestamp"].iloc[-1]}')

# Parse timestamps
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f'\nTest start: {df["timestamp"].min()}')
print(f'Test end:   {df["timestamp"].max()}')
print(f'Duration:   {(df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 60:.1f} minutes')
