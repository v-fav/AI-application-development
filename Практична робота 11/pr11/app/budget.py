"""Денний бюджет у USD; стан зберігається атомарно на диску."""
import os,json,threading,datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
DAILY_USD=max(0,float(os.getenv('BUDGET_DAILY_USD','1.00'))); WARN_AT=min(1,max(0,float(os.getenv('BUDGET_WARN_AT','0.8')))); BUDGET_FILE=os.getenv('BUDGET_FILE','state/budget.json'); _lock=threading.RLock(); _reserved=0.0
class BudgetExceeded(Exception):
    def __init__(self,spent,limit,resets_at):
        self.spent=spent; self.limit=limit; self.resets_at=resets_at
        super().__init__(f'Денний бюджет вичерпано: {spent:.4f} з {limit:.2f} USD. Ліміт відновиться {resets_at}.')
def _day():return datetime.date.today().isoformat()
def _reset():return (datetime.datetime.combine(datetime.date.today()+datetime.timedelta(days=1),datetime.time.min).astimezone().isoformat())
def _read():
    try:
        d=json.loads(Path(BUDGET_FILE).read_text(encoding='utf-8'))
        return float(d.get('spent',0)) if d.get('day')==_day() else 0.0
    except (OSError,ValueError,TypeError):return 0.0
def _write(v):
    p=Path(BUDGET_FILE); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp'); t.write_text(json.dumps({'day':_day(),'spent':round(v,8)}),encoding='utf-8'); os.replace(t,p)
def settings():return {'daily_usd':DAILY_USD,'warn_at':WARN_AT,'file':BUDGET_FILE}
def check(expected_cost=None):
    with _lock:
        spent=_read()
        if spent+_reserved>=DAILY_USD:raise BudgetExceeded(spent,DAILY_USD,_reset())
        if expected_cost is not None and spent+_reserved+max(0,expected_cost)>DAILY_USD:raise BudgetExceeded(spent,DAILY_USD,_reset())
def record(cost):
    global _reserved
    if cost is None:return
    with _lock:_write(_read()+max(0,float(cost)))
def status():
    with _lock:
        spent=_read(); return {'day':_day(),'spent':round(spent,8),'limit':DAILY_USD,'remaining':round(max(0,DAILY_USD-spent),8),'warning':DAILY_USD>0 and spent>=DAILY_USD*WARN_AT,'exceeded':spent>=DAILY_USD,'resets_at':_reset()}
