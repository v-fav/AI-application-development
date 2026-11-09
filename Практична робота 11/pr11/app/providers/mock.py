"""Підставний потоковий провайдер для перевірок без API-ключа."""
import os,random,time,threading
from typing import Iterator
from dotenv import load_dotenv
from .base import Chunk,ProviderError,Request,Result
load_dotenv()
class MockProvider:
    name='mock'; model='mock-1'
    def _reply(self,request):
        from .. import samples
        s=samples.get(str(request.metadata.get('request_id','')))
        if s and s.get('reply_example'):return str(s['reply_example'])
        for m in reversed(request.messages):
            if m.get('role')=='user':return 'Добрий день! Дякуємо за звернення. Оператор уточнить зазначену інформацію.'
        return 'Добрий день! Дякуємо за звернення до служби підтримки «Сузірʼя».'
    def stream(self,request)->Iterator[Chunk]:
        stop=request.metadata.get('_stop_event'); rate=float(request.metadata.get('mock_failure_rate',os.getenv('MOCK_FAILURE_RATE','0'))); kind=str(request.metadata.get('mock_failure_kind',os.getenv('MOCK_FAILURE_KIND','unavailable')))
        failure=kind if random.random()<max(0,min(1,rate)) else None
        first=max(0,int(os.getenv('MOCK_FIRST_DELAY_MS','600')))/1000
        delay=max(0,int(os.getenv('MOCK_DELAY_MS','40')))/1000
        if failure=='timeout':first=max(first,float(os.getenv('LLM_TIMEOUT','20'))+1)
        end=time.monotonic()+first
        while time.monotonic()<end:
            if stop and stop.is_set():raise ProviderError('cancelled','Генерацію скасовано.')
            time.sleep(min(.03,end-time.monotonic()))
        if failure in ('unavailable','auth','rate_limit','timeout'):
            raise ProviderError(failure,{'unavailable':'AI-сервіс тимчасово недоступний.','auth':'Перевірте API-ключ.','rate_limit':'Сервіс тимчасово перевантажений.','timeout':'AI-сервіс не відповів вчасно.'}[failure],retryable=failure!='auth',retry_after=1.0 if failure=='rate_limit' else None)
        text=self._reply(request); words=text.split(); pieces=[]; cur=''
        for w in words:
            if cur and len(cur)+len(w)+1>34:pieces.append(cur);cur=w
            else:cur=(cur+' '+w).strip()
        if cur:pieces.append(cur)
        for i,p in enumerate(pieces,1):
            if stop and stop.is_set():raise ProviderError('cancelled','Генерацію скасовано.')
            if i>1:time.sleep(delay)
            if failure=='mid_stream' and i>3:raise ProviderError('unavailable','Потік відповіді перервався.',retryable=True)
            yield Chunk(p+(' ' if i<len(pieces) else ''))
        inp=max(1,sum(len(str(m.get('content',''))) for m in request.messages)//2); out=max(1,len(text)//2)
        yield Chunk('',usage={'prompt_tokens':inp,'completion_tokens':out,'total_tokens':inp+out},finish_reason='stop')
    def complete(self,request):
        text=''; usage=None; finish=None
        for c in self.stream(request):text+=c.text;usage=c.usage or usage;finish=c.finish_reason or finish
        return Result(text,self.name,self.model,usage,finish)
