"""Єдиний шлюз до провайдерів: retries, cache, budget і JSON-журнал."""
import os,logging,threading,time,queue,random
from dataclasses import replace
from dotenv import load_dotenv
from .providers.base import ProviderError,Request,Result
from . import cache,budget,costs
load_dotenv(); log=logging.getLogger('app.gateway')
PROVIDER=os.getenv('LLM_PROVIDER','mock').strip().lower(); FALLBACK=os.getenv('LLM_FALLBACK','none').strip(); BASE_URL=os.getenv('LLM_BASE_URL',''); API_KEY=os.getenv('LLM_API_KEY','')
TIMEOUT=max(.1,float(os.getenv('LLM_TIMEOUT','20'))); IDLE_TIMEOUT=max(.1,float(os.getenv('LLM_IDLE_TIMEOUT','8'))); TOTAL_TIMEOUT=max(.1,float(os.getenv('LLM_TOTAL_TIMEOUT','45'))); MAX_ATTEMPTS=max(1,int(os.getenv('LLM_MAX_ATTEMPTS','3'))); BACKOFF_BASE=max(0,float(os.getenv('LLM_BACKOFF_BASE','.5'))); BACKOFF_MAX=max(BACKOFF_BASE,float(os.getenv('LLM_BACKOFF_MAX','8'))); MAX_CONCURRENT=max(1,int(os.getenv('LLM_MAX_CONCURRENT','2'))); MODEL=os.getenv('LLM_MODEL',''); TEMPERATURE=float(os.getenv('LLM_TEMPERATURE','.3')); MAX_TOKENS=max(1,int(os.getenv('LLM_MAX_TOKENS','600')))
_SEMAPHORE=threading.BoundedSemaphore(MAX_CONCURRENT)
def settings():return {'provider':PROVIDER,'fallback':FALLBACK,'model':MODEL,'timeout':TIMEOUT,'idle_timeout':IDLE_TIMEOUT,'total_timeout':TOTAL_TIMEOUT,'max_attempts':MAX_ATTEMPTS,'max_concurrent':MAX_CONCURRENT}
def make_provider(name):
    if name.lower()=='mock':
        from .providers.mock import MockProvider
        return MockProvider()
    from .providers.openai_compat import OpenAICompatProvider
    return OpenAICompatProvider(BASE_URL,API_KEY,MODEL if name.lower()=='openai' else name,TIMEOUT)
def _attempt(provider,request,deadline,is_cancelled,on_delta,on_status,number,fallback):
    started=time.monotonic(); remaining=deadline-started
    if remaining<=0:raise ProviderError('timeout','Вичерпано загальний час очікування.',retryable=True)
    if is_cancelled():raise ProviderError('cancelled','Генерацію скасовано.')
    on_status({'status':'running','attempt':number,'provider':provider.name,'fallback':fallback,'detail':'очікування вільного каналу'})
    qstart=time.monotonic()
    if not _SEMAPHORE.acquire(timeout=min(remaining,TIMEOUT)):raise ProviderError('unavailable','Усі канали AI-сервісу зайняті.',retryable=True)
    queue_ms=(time.monotonic()-qstart)*1000; stop=threading.Event(); metadata=dict(request.metadata); metadata['_stop_event']=stop; req=replace(request,metadata=metadata)
    messages=queue.Queue(); attempt_start=time.monotonic()
    def worker():
        stream=None
        try:
            stream=provider.stream(req)
            for chunk in stream:
                if stop.is_set():break
                messages.put(('chunk',chunk))
            messages.put(('end',None))
        except ProviderError as e:messages.put(('error',e))
        except Exception as e:messages.put(('error',ProviderError('unknown','Непередбачена помилка провайдера.',detail=repr(e))))
        finally:
            if stream is not None:
                close=getattr(stream,'close',None)
                if callable(close):
                    try:close()
                    except Exception:pass
            _SEMAPHORE.release()
    threading.Thread(target=worker,name='pr11-provider',daemon=True).start()
    text='';usage=None;finish=None;first=None;last=time.monotonic();error=None
    try:
        while True:
            if is_cancelled():stop.set();raise ProviderError('cancelled','Генерацію скасовано.')
            now=time.monotonic(); wait=min(deadline-now,TIMEOUT-(now-attempt_start),IDLE_TIMEOUT-(now-last) if first is not None else TIMEOUT)
            if wait<=0:stop.set();raise ProviderError('timeout','Перевищено час очікування відповіді.',retryable=True)
            try:kind,value=messages.get(timeout=wait)
            except queue.Empty:stop.set();raise ProviderError('timeout','Потік відповіді не надіслав фрагмент вчасно.',retryable=True)
            if kind=='error':error=value;raise value
            if kind=='end':break
            c=value
            if c.usage:usage=c.usage
            if c.finish_reason:finish=c.finish_reason
            if c.text:
                if first is None:first=time.monotonic();on_status({'status':'running','attempt':number,'provider':provider.name,'fallback':fallback,'first_token':first-attempt_start,'detail':f'перший фрагмент за {first-attempt_start:.2f} с'})
                last=time.monotonic();text+=c.text;on_delta(c.text)
        if not text.strip():raise ProviderError('unavailable','AI-сервіс завершив відповідь без тексту.',retryable=True)
        return Result(text,provider.name,getattr(provider,'model',request.model or MODEL or 'unknown'),usage,finish,time.monotonic()-attempt_start),{'queue_ms':round(queue_ms,1),'first_token_ms':round((first-attempt_start)*1000,1) if first else None,'duration_ms':round((time.monotonic()-attempt_start)*1000,1)}
    except ProviderError as exc:
        exc.partial_text=text
        exc.usage=usage
        exc.timing={'queue_ms':round(queue_ms,1),'first_token_ms':round((first-attempt_start)*1000,1) if first else None,'duration_ms':round((time.monotonic()-attempt_start)*1000,1)}
        raise
    finally:
        if error is not None:stop.set()
def generate(request,on_delta,on_status,is_cancelled):
    started=time.monotonic();deadline=started+TOTAL_TIMEOUT; draft_id=request.metadata.get('draft_id'); request_id=request.metadata.get('http_request_id'); client=request.metadata.get('client','anonymous'); version=request.metadata.get('prompt_version','unknown'); key=cache.make_key(request,version); fresh=bool(request.metadata.get('fresh'))
    if fresh:
        cache.bypassed(); log.info('cache.bypass',extra={'draft_id':draft_id,'cache_key':key,'prompt_version':version})
    elif cache.ENABLED:
        hit=cache.get(key)
        if hit:
            hit.cached=True;on_status({'status':'running','attempt':0,'provider':hit.provider,'fallback':False,'cached':True,'detail':'відповідь отримано з кешу'});on_delta(hit.text)
            log.info('cache.hit',extra={'draft_id':draft_id,'cache_key':key,'cached':True,'prompt_version':version})
            return replace(hit,elapsed=0.0,cached=True)
        log.info('cache.miss',extra={'draft_id':draft_id,'cache_key':key,'prompt_version':version})
    primary=make_provider(PROVIDER); last=ProviderError('unknown','Не вдалося отримати відповідь.'); emitted=False;attempts=0
    def delta(v):
        nonlocal emitted
        emitted=True;on_delta(v)
    providers=[(primary,False)]
    for number in range(1,MAX_ATTEMPTS+1):
        attempts=number
        if is_cancelled():raise ProviderError('cancelled','Генерацію скасовано.')
        try:budget.check()
        except budget.BudgetExceeded as e:raise ProviderError('budget',str(e),retryable=False,detail=f'resets_at={e.resets_at}') from e
        p=primary; t=time.monotonic(); meta={'request_id':request_id,'draft_id':draft_id,'client':client,'provider':p.name,'model':request.model or MODEL,'prompt_version':version,'attempt':number,'fallback':False,'cache_key':key,'status':'started','queue_ms':None,'first_token_ms':None,'duration_ms':None}
        log.info('llm.attempt',extra=meta)
        try:
            result,timing=_attempt(p,request,deadline,is_cancelled,delta,on_status,number,False)
            cost,estimated=costs.estimate(result.usage,provider=p.name,model=result.model,text=result.text);result=replace(result,cost=cost,cost_estimated=estimated,cached=False)
            budget.record(cost);meta.update(timing);meta.update({'status':'success','model':result.model,'prompt_tokens':(result.usage or {}).get('prompt_tokens'),'completion_tokens':(result.usage or {}).get('completion_tokens'),'cost':cost,'cost_estimated':estimated,'cached':False})
            log.info('llm.attempt.done',extra=meta)
            cache.put(key,result)
            if budget.status()['warning']:log.warning('budget.warning',extra={'draft_id':draft_id,**budget.status()})
            return replace(result,elapsed=time.monotonic()-started)
        except ProviderError as e:
            last=e
            failed_cost,failed_estimated=costs.estimate(getattr(e,'usage',None),provider=p.name,model=request.model,text=getattr(e,'partial_text',''))
            if failed_cost is not None: budget.record(failed_cost)
            meta.update(getattr(e,'timing',{}))
            meta.update({'status':'failed','kind':e.kind,'duration_ms':round((time.monotonic()-t)*1000,1),'cost':failed_cost,'cost_estimated':failed_estimated,'cached':False,'prompt_tokens':(getattr(e,'usage',None) or {}).get('prompt_tokens'),'completion_tokens':(getattr(e,'usage',None) or {}).get('completion_tokens')})
            log.warning('llm.attempt.failed' if number<MAX_ATTEMPTS and e.retryable else 'llm.attempt.error',extra=meta)
            if e.kind in ('cancelled','budget') or emitted or not e.retryable or number>=MAX_ATTEMPTS:break
            delay=e.retry_after if e.retry_after is not None else min(BACKOFF_MAX,BACKOFF_BASE*2**(number-1))*random.uniform(.5,1.5)
            if time.monotonic()+delay>=deadline:break
            on_status({'status':'running','attempt':number,'provider':p.name,'fallback':False,'detail':f'повтор через {delay:.1f} с'})
            end=time.monotonic()+delay
            while time.monotonic()<end:
                if is_cancelled():raise ProviderError('cancelled','Генерацію скасовано.')
                time.sleep(min(.05,end-time.monotonic()))
    if not emitted and FALLBACK.lower() not in ('','none',PROVIDER) and time.monotonic()<deadline:
        try:
            backup=make_provider(FALLBACK);budget.check();on_status({'status':'running','attempt':attempts,'provider':backup.name,'fallback':True,'detail':'використовується запасний провайдер'})
            r,timing=_attempt(backup,request,deadline,is_cancelled,delta,on_status,attempts,True);cost,est=costs.estimate(r.usage,provider=backup.name,model=r.model,text=r.text);r=replace(r,cost=cost,cost_estimated=est);budget.record(cost);cache.put(key,r);log.warning('llm.fallback',extra={'draft_id':draft_id,'request_id':request_id,'provider':backup.name,'model':r.model,'prompt_version':version,'attempt':attempts,'fallback':True,'duration_ms':timing['duration_ms'],'cost':cost,'cost_estimated':est});return replace(r,elapsed=time.monotonic()-started)
        except budget.BudgetExceeded as e:last=ProviderError('budget',str(e),retryable=False)
        except ProviderError as e:last=e
    log.error('llm.failed',extra={'draft_id':draft_id,'request_id':request_id,'attempts':attempts,'kind':last.kind,'provider':PROVIDER,'prompt_version':version})
    raise last
