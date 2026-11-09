"""Ковзне вікно обмеження частоти для кожного клієнта."""
import os,threading,time
from collections import defaultdict,deque
from dotenv import load_dotenv
load_dotenv();PER_MINUTE=max(1,int(os.getenv('RATE_LIMIT_PER_MINUTE','10')));WINDOW_SECONDS=60.0;_requests=defaultdict(deque);_lock=threading.Lock()
class RateLimited(Exception):
    def __init__(self,retry_after):self.retry_after=max(.1,retry_after);super().__init__(f'ліміт вичерпано, повторіть через {self.retry_after:.1f} с')
def settings():return {'per_minute':PER_MINUTE,'window_seconds':WINDOW_SECONDS}
def acquire(client):
    key=(str(client or 'anonymous').strip() or 'anonymous')[:128];now=time.monotonic();cut=now-WINDOW_SECONDS
    with _lock:
        h=_requests[key]
        while h and h[0]<=cut:h.popleft()
        if len(h)>=PER_MINUTE:raise RateLimited(WINDOW_SECONDS-(now-h[0]))
        h.append(now)
        if len(_requests)>1000:
            for k in list(_requests):
                if not _requests[k]:_requests.pop(k,None)
