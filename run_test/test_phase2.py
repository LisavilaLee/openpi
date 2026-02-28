import time
import os
import sys
print('=== Phase 2: HF & Config ===')
t0 = time.time()
try:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from openpi.training import config
    print(f'Import config: {time.time()-t0:.2f}s')
except Exception as e:
    print(f'Import config failed: {e}')

t0 = time.time()
try:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_LEROBOT_HOME'] = '/storages/liweile/.cache/huggingface/lerobot'
    import lerobot
    print(f'Import lerobot: {time.time()-t0:.2f}s')
    cache_dir = os.environ['HF_LEROBOT_HOME']
    print(f'Listing {cache_dir}...')
    if os.path.exists(cache_dir):
        files = os.listdir(cache_dir)
        print(f'Cache dir contains {len(files)} items. Time: {time.time()-t0:.2f}s')
    else:
        print('Cache dir not found.')
except Exception as e:
    print(f'HF check failed: {e}')
