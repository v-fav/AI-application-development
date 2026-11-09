"""Фонові задачі й події чернеток."""
import datetime,threading,time,uuid,logging
from dataclasses import dataclass,field
from typing import Iterator
from . import drafting,gateway,telemetry
from .providers.base import ProviderError
TERMINAL={'done','failed','cancelled'}; log=logging.getLogger('app.jobs')
@dataclass
class Draft:
    draft_id:str; request_id:str; client:str; status:str='queued'; text:str=''; attempts:int=0; provider:str|None=None; model:str|None=None; fallback:bool=False; error:dict|None=None; usage:dict|None=None; elapsed:dict=field(default_factory=dict); warnings:list=field(default_factory=list); created_at:str=''; updated_at:str=''; events:list=field(default_factory=list); cost:float|None=None; cost_estimated:bool=False; cached:bool=False
_drafts={};_order=[];_cancel={};_lock=threading.Lock();_changed=threading.Condition(_lock);_MAX=200
def _now():return datetime.datetime.now().isoformat(timespec='seconds')
def _event(d,event,data=None):d.events.append({'id':len(d.events)+1,'event':event,'data':data or {}});d.updated_at=_now();_changed.notify_all()
def new_draft(request_id,client):
    d=Draft(uuid.uuid4().hex[:8],request_id,client,created_at=_now(),updated_at=_now())
    with _changed:
        _drafts[d.draft_id]=d;_order.insert(0,d.draft_id)
        while len(_order)>_MAX:
            old=_order[-1]
            if _drafts[old].status not in TERMINAL:break
            _order.pop();_drafts.pop(old,None)
    return d
def get(draft_id):
    with _lock:return _drafts.get(draft_id)
def list_drafts(limit=50):
    with _lock:return [_drafts[i] for i in _order[:limit] if i in _drafts]
def emit(d,event,data=None):
    with _changed:_event(d,event,data)
def subscribe(draft_id,since=0,heartbeat=15.0):
    while True:
        with _changed:
            d=_drafts.get(draft_id)
            if d is None:return
            pending=d.events[since:]
            if not pending:
                if d.status in TERMINAL:return
                _changed.wait(timeout=heartbeat);pending=d.events[since:]
        if not pending:yield None;continue
        for e in pending:yield e
        since+=len(pending)
def _run(d,sample,cancel,fresh,http_id):
    started=time.monotonic();first=None
    def status(info):
        with _changed:
            if d.status in TERMINAL or cancel.is_set():return
            d.status=info.get('status','running');d.attempts=max(d.attempts,int(info.get('attempt',0)));d.provider=info.get('provider') or d.provider;d.fallback=bool(info.get('fallback',d.fallback));_event(d,'status',info)
    def delta(value):
        nonlocal first
        with _changed:
            if d.status in TERMINAL or cancel.is_set():return
            if first is None:first=time.monotonic()
            d.text+=value;_event(d,'delta',{'text':value})
    try:
        with _changed:d.status='running';_event(d,'status',{'status':'running','attempt':0,'provider':gateway.PROVIDER,'detail':'фонову генерацію розпочато'})
        req=drafting.build_request(sample);req.metadata.update({'draft_id':d.draft_id,'http_request_id':http_id,'client':d.client,'fresh':fresh})
        result=gateway.generate(req,delta,status,cancel.is_set)
        with _changed:
            if d.status in TERMINAL or cancel.is_set():return
            d.text=result.text;d.provider=result.provider;d.model=result.model;d.usage=result.usage;d.cost=result.cost;d.cost_estimated=result.cost_estimated;d.cached=result.cached
            d.elapsed={'first_token':max(0,first-started) if first else 0,'total':float(result.elapsed or time.monotonic()-started)};d.warnings=drafting.check(result.text,sample);d.status='done'
            _event(d,'done',{'text':d.text,'provider':d.provider,'model':d.model,'usage':d.usage,'elapsed':d.elapsed,'attempts':d.attempts,'fallback':d.fallback,'warnings':d.warnings,'cost':d.cost,'cost_estimated':d.cost_estimated,'cached':d.cached})
        log.info('draft.done',extra={'draft_id':d.draft_id,'request_id':http_id,'sample_id':d.request_id,'client_id_hash':__import__('hashlib').sha256(d.client.encode()).hexdigest()[:12],'duration_ms':round((time.monotonic()-started)*1000,1),'attempts':d.attempts,'cost':d.cost,'cached':d.cached})
    except ProviderError as e:
        with _changed:
            if d.status in TERMINAL:return
            d.status='cancelled' if e.kind=='cancelled' else 'failed';d.error={'kind':e.kind,'message':e.message,'retryable':e.retryable};_event(d,'cancelled' if e.kind=='cancelled' else 'error',d.error|{'request_id':http_id})
        log.warning('draft.failed',extra={'draft_id':d.draft_id,'request_id':http_id,'kind':e.kind,'duration_ms':round((time.monotonic()-started)*1000,1)})
    except Exception:
        log.exception('draft.unexpected',extra={'draft_id':d.draft_id,'request_id':http_id})
        with _changed:
            if d.status not in TERMINAL:d.status='failed';d.error={'kind':'unknown','message':'Внутрішня помилка генерації.','retryable':True};_event(d,'error',d.error|{'request_id':http_id})
    finally:
        with _lock:_cancel.pop(d.draft_id,None)
def start(sample,client,fresh=False):
    if not sample or not sample.get('id'):raise ValueError('Невідоме звернення')
    http_id=telemetry.request_id_var.get() or telemetry.new_request_id();d=new_draft(str(sample['id']),str(client or 'anonymous'));ev=threading.Event()
    with _changed:_cancel[d.draft_id]=ev;_event(d,'status',{'status':'queued','attempt':0,'provider':gateway.PROVIDER,'detail':'чернетку додано до черги'})
    thread=threading.Thread(target=_run,args=(d,sample,ev,fresh,http_id),name='draft-'+d.draft_id,daemon=True)
    try:thread.start()
    except Exception:
        with _changed:d.status='failed';d.error={'kind':'unavailable','message':'Не вдалося запустити фонову задачу.','retryable':True};_event(d,'error',d.error);_cancel.pop(d.draft_id,None)
        raise
    return d
def cancel(draft_id):
    with _changed:
        d=_drafts.get(draft_id)
        if d is None:raise KeyError(draft_id)
        if d.status in TERMINAL:raise ValueError('Чернетку вже завершено')
        ev=_cancel.get(draft_id)
        if ev:ev.set()
        d.status='cancelled';_event(d,'cancelled',{'text':d.text});return d
