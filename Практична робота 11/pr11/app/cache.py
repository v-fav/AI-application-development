"""Персистентний exact-cache з TTL та обмеженням розміру."""
import os,json,time,hashlib,threading
from pathlib import Path
from dataclasses import asdict
from dotenv import load_dotenv
from .providers.base import Request,Result
load_dotenv()
ENABLED=os.getenv('CACHE_ENABLED','1')=='1'; TTL_SECONDS=max(0,float(os.getenv('CACHE_TTL_SECONDS','3600'))); MAX_ENTRIES=max(1,int(os.getenv('CACHE_MAX_ENTRIES','500'))); CACHE_FILE=os.getenv('CACHE_FILE','state/cache.json')
_lock=threading.RLock(); _hits=0; _misses=0; _bypassed=0

def settings(): return {'enabled':ENABLED,'ttl_seconds':TTL_SECONDS,'max_entries':MAX_ENTRIES,'file':CACHE_FILE or None}
def _load():
    if not CACHE_FILE:return {}
    try:
        with open(CACHE_FILE,encoding='utf-8') as f:return json.load(f)
    except (OSError,ValueError):return {}
def _save(data):
    if not CACHE_FILE:return
    p=Path(CACHE_FILE); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8'); os.replace(tmp,p)
def make_key(request:Request,prompt_version:str)->str:
    data={'provider':os.getenv('LLM_PROVIDER','mock'),'model':request.model or os.getenv('LLM_MODEL',''),'prompt_version':prompt_version,'messages':request.messages,'temperature':request.temperature,'max_tokens':request.max_tokens}
    raw=json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(raw.encode()).hexdigest()
def get(key):
    global _hits,_misses
    if not ENABLED:return None
    with _lock:
        data=_load(); item=data.get(key)
        if not item: _misses+=1; return None
        if time.time()-float(item.get('saved',0))>TTL_SECONDS:
            data.pop(key,None); _save(data); _misses+=1; return None
        _hits+=1; d=item['result']; return Result(text=d['text'],provider=d['provider'],model=d['model'],usage=d.get('usage'),finish_reason=d.get('finish_reason'),elapsed=0.0,cost=d.get('cost'),cost_estimated=d.get('cost_estimated',False),cached=True)
def put(key,result):
    if not ENABLED:return
    with _lock:
        data=_load(); data[key]={'saved':time.time(),'result':asdict(result)}
        if len(data)>MAX_ENTRIES:
            data=dict(sorted(data.items(),key=lambda x:x[1].get('saved',0))[-MAX_ENTRIES:])
        _save(data)
def bypassed():
    global _bypassed
    with _lock:_bypassed+=1
def stats():
    with _lock:
        data=_load(); now=time.time(); valid={k:v for k,v in data.items() if now-float(v.get('saved',0))<=TTL_SECONDS}
        if len(valid)!=len(data):_save(valid)
        return {'hits':_hits,'misses':_misses,'bypassed':_bypassed,'entries':len(valid)}
