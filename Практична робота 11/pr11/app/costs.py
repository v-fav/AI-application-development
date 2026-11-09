"""Розрахунок вартості викликів за usage або приблизною кількістю токенів."""
import os
from dotenv import load_dotenv
load_dotenv()
PRICE_INPUT=float(os.getenv('LLM_PRICE_INPUT_PER_1M','0.30'))
PRICE_OUTPUT=float(os.getenv('LLM_PRICE_OUTPUT_PER_1M','2.50'))
PRICE_CACHED_INPUT=float(os.getenv('LLM_PRICE_CACHED_INPUT_PER_1M') or PRICE_INPUT*0.25)
MOCK_BILLABLE=os.getenv('MOCK_BILLABLE','1')=='1'
def settings():
    return {'input_per_1m':PRICE_INPUT,'output_per_1m':PRICE_OUTPUT,'cached_input_per_1m':PRICE_CACHED_INPUT,'mock_billable':MOCK_BILLABLE,'currency':'USD'}
def estimate(usage, *, provider=None, model=None, text=None):
    if provider=='mock' and not MOCK_BILLABLE: return 0.0, False
    if isinstance(usage,dict) and ('prompt_tokens' in usage or 'completion_tokens' in usage):
        inp=max(0,int(usage.get('prompt_tokens') or 0)); out=max(0,int(usage.get('completion_tokens') or 0))
        details=usage.get('prompt_tokens_details') or {}; cached=max(0,int(details.get('cached_tokens') or 0))
        cached=min(cached,inp)
        value=((inp-cached)*PRICE_INPUT+cached*PRICE_CACHED_INPUT+out*PRICE_OUTPUT)/1_000_000
        return round(value,8),False
    if text is None: return None,True
    # Консервативна оцінка: приблизно 2 символи на токен для українського тексту.
    out=max(1,(len(text)+1)//2)
    inp=0
    return round((inp*PRICE_INPUT+out*PRICE_OUTPUT)/1_000_000,8),True
