import os
import time

files = [
    'src/models/dataset.py',
    'src/models/ranker.py', 
    'src/models/candidate_generation.py',
    'src/models/serving.py',
    'models/artifacts/purchase_ranker.pkl'
]

print('=' * 80)
print('FILE MODIFICATION TIMES')
print('=' * 80)
print()

for f in files:
    if os.path.exists(f):
        mtime = os.path.getmtime(f)
        print(f'{f:50s} {time.ctime(mtime)}')
    else:
        print(f'{f:50s} NOT FOUND')

print()
print('=' * 80)
print('ANALYSIS')
print('=' * 80)

model_time = os.path.getmtime('models/artifacts/purchase_ranker.pkl')
print(f'\nModel trained at: {time.ctime(model_time)}')
print(f'Model timestamp: {model_time}')

print('\nSource files modified AFTER model training:')
for f in files[:-1]:  # Exclude model file
    if os.path.exists(f):
        file_time = os.path.getmtime(f)
        if file_time > model_time:
            print(f'  [NEWER] {f} - {time.ctime(file_time)}')
        else:
            print(f'  [OLDER] {f} - {time.ctime(file_time)}')
