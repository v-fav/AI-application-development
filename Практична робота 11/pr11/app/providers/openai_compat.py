"""Адаптер OpenAI-compatible Chat Completions API."""
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
import threading
from typing import Iterator
from .base import Chunk,ProviderError,Request,Result

def _retry_after(headers):
    if not headers:return None
    v=headers.get('retry-after') or headers.get('Retry-After')
    if v:
        try:return max(0,float(v))
        except (ValueError,TypeError):
            try:
                d=parsedate_to_datetime(v); d=d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d
                return max(0,(d-datetime.now(timezone.utc)).total_seconds())
            except Exception:pass
    return None
class OpenAICompatProvider:
    name='openai'
    def __init__(self,base_url,api_key,model,timeout):self.base_url=base_url;self.api_key=api_key;self.model=model;self.timeout=max(.1,float(timeout));self._client=None
    def _get_client(self):
        if not self.api_key:raise ProviderError('auth','Не вказано LLM_API_KEY.',retryable=False)
        if self._client is None:
            try:
                from openai import OpenAI
                self._client=OpenAI(base_url=self.base_url or None,api_key=self.api_key,timeout=self.timeout,max_retries=0)
            except ImportError as e:raise ProviderError('unavailable','Не встановлено пакет openai.',detail=repr(e)) from e
        return self._client
    @staticmethod
    def _usage(v):
        if v is None:return None
        if hasattr(v,'model_dump'):return v.model_dump(exclude_none=True)
        return v if isinstance(v,dict) else None
    @staticmethod
    def _error(e):
        try:import openai
        except ImportError:return ProviderError('unknown','Помилка AI-сервісу.',detail=repr(e))
        response=getattr(e,'response',None); retry=_retry_after(getattr(response,'headers',None)); detail=f'{type(e).__name__}: {e}'
        if isinstance(e,openai.APITimeoutError):return ProviderError('timeout','AI-сервіс не відповів вчасно.',retryable=True,detail=detail)
        if isinstance(e,openai.RateLimitError):return ProviderError('rate_limit','AI-сервіс обмежив кількість запитів.',retryable=True,retry_after=retry,detail=detail)
        if isinstance(e,openai.AuthenticationError):return ProviderError('auth','Перевірте API-ключ.',detail=detail)
        if isinstance(e,openai.BadRequestError):return ProviderError('bad_request','AI-сервіс відхилив запит.',detail=detail)
        if isinstance(e,openai.APIConnectionError):return ProviderError('unavailable','Немає зʼєднання з AI-сервісом.',retryable=True,detail=detail)
        if isinstance(e,openai.APIStatusError):
            code=getattr(e,'status_code',0)
            if code==429:return ProviderError('rate_limit','AI-сервіс обмежив кількість запитів.',retryable=True,retry_after=retry,detail=detail)
            if code in (401,403):return ProviderError('auth','AI-сервіс не прийняв ключ.',detail=detail)
            if code>=500:return ProviderError('unavailable','AI-сервіс тимчасово недоступний.',retryable=True,detail=detail)
            return ProviderError('bad_request','AI-сервіс відхилив запит.',detail=detail)
        return ProviderError('unknown','Непередбачена помилка AI-сервісу.',detail=detail)
    def stream(self,request)->Iterator[Chunk]:
        model=request.model or self.model
        try:raw=self._get_client().chat.completions.create(model=model,messages=request.messages,temperature=request.temperature,max_tokens=request.max_tokens,stream=True,stream_options={'include_usage':True})
        except Exception as e:raise self._error(e) from e
        lock=threading.Lock(); closed=False
        def close():
            nonlocal closed
            with lock:
                if not closed:
                    closed=True
                    fn=getattr(raw,'close',None)
                    if callable(fn):
                        try:fn()
                        except Exception:pass
        request.metadata['_close_stream']=close; stop=request.metadata.get('_stop_event')
        try:
            for part in raw:
                if stop is not None and stop.is_set():break
                choices=getattr(part,'choices',None) or []
                if choices:
                    c=choices[0]; d=getattr(c,'delta',None); content=getattr(d,'content',None) if d else None
                    yield Chunk(content or '',self._usage(getattr(part,'usage',None)),getattr(c,'finish_reason',None))
                else:
                    usage=self._usage(getattr(part,'usage',None))
                    if usage:yield Chunk('',usage)
        except Exception as e:raise self._error(e) from e
        finally:close()
    def complete(self,request):
        text='';usage=None;finish=None
        for c in self.stream(request):text+=c.text;usage=c.usage or usage;finish=c.finish_reason or finish
        return Result(text,self.name,request.model or self.model,usage,finish)
