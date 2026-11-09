"""Аналіз журналу: повільні, невдалі й повторні виклики, витрати, кеш.

Запуск із папки pr11:

    python eval/analyze.py                       # logs/app.jsonl, усі розділи
    python eval/analyze.py logs/app.jsonl --since 2026-10-23 --section slow

Читання файлу, відбір за датою і допоміжні функції готові. Розділи
звіту — заготовки: поля, за якими вони працюють, — ваші, бо назви
подій і полів ви обрали, коли писали записи в шлюзі й фонових задачах.
`summary` працює на будь-якому журналі й показує, що в ньому взагалі є.

Орієнтир для кожного розділу — питання, на яке він відповідає; воно в
докстрінгу. Відповідь — числа й кілька рядків журналу, а не переказ.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path, since: str | None = None) -> list[dict]:
    """Прочитати JSON lines. Рядки, які не розбираються, рахуються й
    пропускаються; `since` — дата або час у форматі ISO, від якого брати
    записи (порівнюється з полем `ts` як рядок)."""
    records: list[dict] = []
    broken = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                broken += 1
                continue
            if since and str(record.get("ts", "")) < since:
                continue
            records.append(record)
    if broken:
        print(f"(пропущено рядків, що не розбираються: {broken})", file=sys.stderr)
    return records


def percentile(values: list[float], p: float) -> float | None:
    """p-й процентиль (0 ≤ p ≤ 1) за найближчим рангом; `None` для
    порожнього списку."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def group(records: list[dict], key: str) -> dict:
    """Згрупувати записи за значенням поля."""
    groups: dict = defaultdict(list)
    for record in records:
        groups[record.get(key)].append(record)
    return groups


def summary(records: list[dict]) -> None:
    """Що взагалі є в журналі: скільки записів, за який час, яких подій і
    рівнів скільки. Працює на будь-якому журналі."""
    if not records:
        print("записів немає")
        return
    stamps = [r.get("ts", "") for r in records]
    print(f"записів: {len(records)}; від {min(stamps)} до {max(stamps)}")
    print("рівні:", dict(Counter(r.get("level") for r in records)))
    print("події:")
    for event, count in Counter(r.get("event") for r in records).most_common():
        print(f"  {count:>6}  {event}")


def slow(records: list[dict]) -> None:
    """Які виклики найповільніші й що таке «повільно» в цих числах?

    Розподіл тривалості звертань до провайдера (медіана, p95, максимум),
    а не середнє; десятка найдовших із ідентифікаторами, провайдером,
    моделлю, спробою; де пройшов час — у черзі, до першого фрагмента чи
    в генерації; розріз за провайдером і годиною.
    """
    rows=[r for r in records if r.get('event')=='llm.attempt.done' or r.get('event')=='llm.attempt.failed' or r.get('event')=='llm.attempt.error']
    vals=[float(r.get('duration_ms') or 0) for r in rows]
    if not vals: print('викликів провайдера немає'); return
    import statistics
    print(f"викликів: {len(vals)}; медіана: {statistics.median(vals):.1f} ms; p95: {percentile(vals,.95):.1f} ms; максимум: {max(vals):.1f} ms")
    for r in sorted(rows,key=lambda x:float(x.get('duration_ms') or 0),reverse=True)[:10]: print(f"  {r.get('duration_ms')} ms draft={r.get('draft_id')} attempt={r.get('attempt')} provider={r.get('provider')} first={r.get('first_token_ms')} queue={r.get('queue_ms')}")


def failed(records: list[dict]) -> None:
    """Що й чому не вдалося?

    Невдалі спроби за видом збою; остаточні збої чернеток; частка
    чернеток із невдалою першою спробою, які все ж завершилися успіхом
    (чи окупилися повтори); вартість спроб, які нічого не дали.
    """
    rows=[r for r in records if r.get('event') in ('llm.attempt.failed','llm.attempt.error')]; print('невдалі спроби за видом:',dict(Counter(r.get('kind','unknown') for r in rows)))
    drafts=group(rows,'draft_id'); print('чернеток із невдалими спробами:',len(drafts)); print('вартість невдалих спроб:',sum(float(r.get('cost') or 0) for r in rows))


def repeated(records: list[dict]) -> None:
    """Що ми робили двічі?

    Той самий ключ кешу кілька разів за короткий час (кандидати на кеш
    і питання «чому»); чернетки з кількома спробами (повтори шлюзу);
    клієнти з серіями замовлень за хвилину.
    """
    attempts=[r for r in records if r.get('event','').startswith('llm.attempt')]; by=group(attempts,'draft_id')
    print('чернетки з кількома спробами:')
    for k,v in by.items():
        if k and len(v)>1: print(f'  {k}: {len(v)} спроб')
    keys=Counter(r.get('cache_key') for r in attempts if r.get('cache_key')); print('повторні ключі:',[(k[:12],n) for k,n in keys.items() if n>1][:20])


def spend(records: list[dict]) -> None:
    """Скільки це коштувало?

    Вартість і токени за день, за модель, за клієнта; середня вартість
    чернетки; частка вартості, яка припала на спроби зі збоєм і на
    запасний варіант; скільки записів без вартості або з оцінкою.
    """
    rows=[r for r in records if r.get('event') in ('llm.attempt.done','llm.fallback')]; total=sum(float(r.get('cost') or 0) for r in rows)
    print(f'витрати: ${total:.8f}; викликів: {len(rows)}; оцінок: {sum(bool(r.get("cost_estimated")) for r in rows)}')
    for model,items in group(rows,'model').items(): print(f'  {model}: ${sum(float(x.get("cost") or 0) for x in items):.8f}, викликів {len(items)}')
    print('токени input/output:',sum(int(r.get('prompt_tokens') or 0) for r in rows),sum(int(r.get('completion_tokens') or 0) for r in rows))


def cache_report(records: list[dict]) -> None:
    """Що дав кеш?

    Влучання, промахи, обходи (`fresh`) і частка влучань; скільки
    звертань до провайдера й грошей заощаджено; які звернення
    повторювалися найчастіше.
    """
    hits=sum(r.get('event')=='cache.hit' for r in records); misses=sum(r.get('event')=='cache.miss' for r in records); bypass=sum(r.get('event')=='cache.bypass' for r in records)
    print(f'влучання: {hits}; промахи: {misses}; обходи: {bypass}; частка влучань: {hits/(hits+misses):.1%}' if hits+misses else f'влучання: {hits}; промахи: {misses}; обходи: {bypass}')


SECTIONS = {
    "summary": summary,
    "slow": slow,
    "failed": failed,
    "repeated": repeated,
    "spend": spend,
    "cache": cache_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", default="logs/app.jsonl")
    parser.add_argument("--since", help="брати записи від цієї дати/часу (ISO), наприклад 2026-10-23")
    parser.add_argument("--section", choices=["all", *SECTIONS], default="all")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"журналу немає: {path}", file=sys.stderr)
        return 1
    records = load(path, args.since)

    names = list(SECTIONS) if args.section == "all" else [args.section]
    for name in names:
        print(f"\n=== {name}")
        try:
            SECTIONS[name](records)
        except Exception as exc:
            print(f"(помилка розділу: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
