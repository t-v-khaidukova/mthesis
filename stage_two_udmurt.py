"""
stage_two_udmurt.py
===================
Адаптация stage_two.py для задачи морфологической дизамбигуации удмуртского языка.
Положить в корень проекта рядом с stage_two.py.

Запуск:
    uv run stage_two_udmurt.py \
        --corpus ./eval/corpus.csv \
        --tbox-path ./tbox.json \
        --tbox-interpreter-path ./interpreter.py \
        --model claude-sonnet-4-6

Логика такая же, как в stage_two.py, но заменены 4 функции:
    load_data           → load_udmurt_data
    predict_solar       → predict_solar_udmurt
    calculate_success_metric → calc_morpho_accuracy
    main                → main (адаптирован)
"""

import json
import time
from argparse import ArgumentParser
from functools import cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ── Импортируем всё неизменное из оригинального stage_two ────────────────────
from stage_two import (
    QueryMetrics,
    SummaryStats,
    predict_with_metrics,
    chat_model,
    print_summary_stats,
    get_prediction_method,   # нам нужен только solar-режим
)
from solar.model import TBox
from solar.knowledge_application import predict as predict_solar_core

# ── Наш новый конвертер ───────────────────────────────────────────────────────
from udmurt_corpus_parser import uniparser_to_abox, parse_corpus_file


# =============================================================================
# ИЗМЕНЕНИЕ 1: Загрузка данных
# =============================================================================

def load_udmurt_data(corpus_path: str) -> pd.DataFrame:
    """
    Читает корпусный CSV и возвращает DataFrame с колонками:
        sentence         — текст предложения
        uniparser_output — строка UniParser (леммы TAB теги)
        gold_tag         — эталонный тег (заполняется вручную или оставляется пустым)
        source           — источник (газета, год)

    Эталонная колонка gold_tag может быть пустой — тогда оценка не считается,
    система просто выводит результаты для ручной проверки.
    """
    records = parse_corpus_file(corpus_path)
    rows = []
    for r in records:
        rows.append({
            "sent_id":          r["sent_id"],
            "sentence":         r["sentence"],
            "uniparser_output": r["uniparser_output"],
            "source":           r["source"],
            "gold_tag":         "",   # заполни вручную или загрузи из gold.json
        })
    df = pd.DataFrame(rows)

    # Если рядом лежит gold.json — подгружаем эталоны
    gold_path = Path(corpus_path).with_suffix(".gold.json")
    if gold_path.exists():
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        gold_map = {g["sent_id"]: g.get("gold_tag", "") for g in gold}
        df["gold_tag"] = df["sent_id"].map(gold_map).fillna("")

    return df


# =============================================================================
# ИЗМЕНЕНИЕ 2: Функция предсказания (заменяет predict_solar из stage_two.py)
# =============================================================================

def predict_solar_udmurt(
    model: str,
    task: str,          # JSON-строка с {"sent_id", "sentence", "uniparser_output"}
    debug: bool,
    no_cache: bool,
    tbox_path: str,
    tbox_interpreter_path: str,
) -> str | None:
    """
    Вместо LLM-парсинга текста (FACT_EXTRACTION_PROMPT) используем
    детерминированный конвертер uniparser_to_abox().

    Возвращает строку с выбранным тегом, например 'STEM-PST-3SG',
    или None если дизамбигуация не удалась.
    """
    with open(tbox_path) as f:
        tbox = TBox.model_validate(json.load(f))
    with open(tbox_interpreter_path) as f:
        tbox_interpreter = f.read()

    task_data = json.loads(task)
    sent_id          = task_data["sent_id"]
    sentence         = task_data["sentence"]
    uniparser_output = task_data["uniparser_output"]

    # ── Строим ABox детерминированно (не через LLM) ──────────────────────────
    abox = uniparser_to_abox(sentence, uniparser_output, sent_id)

    if debug:
        print(f"\n[{sent_id}] {sentence}")
        ambiguous = _count_ambiguous_tokens(abox)
        print(f"  Неоднозначных токенов: {ambiguous}")

    # ── Формируем вопрос для predict_solar_core ───────────────────────────────
    # predict_solar_core ожидает (llm, description, question, tbox, interpreter, debug)
    # description = ABox в виде JSON-строки
    # question    = что нужно определить
    description = json.dumps(abox, ensure_ascii=False)
    question    = (
        f"For sentence '{sentence}', resolve morphological ambiguity: "
        f"select the correct candidate analysis for each ambiguous token. "
        f"Return the selected raw tag for each resolved token."
    )

    try:
        result, _ = predict_solar_core(
            chat_model(model, no_cache),
            description,
            question,
            tbox,
            tbox_interpreter,
            debug,
        )
        # result — строка с ответом LLM (тег или JSON)
        return str(result).strip() if result else None
    except Exception as e:
        if debug:
            print(f"  [!] Ошибка predict_solar_core: {e}")
        return None


def _count_ambiguous_tokens(abox: dict) -> int:
    token_candidates: dict = {}
    for a in abox["assertions"]:
        if a["predicate"] == "candidateOf":
            tok = a["args"][1]
            token_candidates[tok] = token_candidates.get(tok, 0) + 1
    return sum(1 for v in token_candidates.values() if v > 1)


# =============================================================================
# ИЗМЕНЕНИЕ 3: Метрика оценки (заменяет calculate_success_metric)
# =============================================================================

def calc_morpho_accuracy(
    true_tags: list[str],
    predictions: list[str | None],
) -> tuple[float, int]:
    """
    Exact-match accuracy по морфологическому тегу.
    Пропускает пары, где prediction=None (система не смогла ответить).
    """
    valid_pairs = [
        (true, pred)
        for true, pred in zip(true_tags, predictions)
        if pred is not None and true != ""
    ]
    if not valid_pairs:
        return 0.0, 0

    correct = sum(1 for true, pred in valid_pairs if true in pred)
    return correct / len(valid_pairs), len(valid_pairs)


def calculate_summary_stats_udmurt(
    metrics: list[QueryMetrics],
    true_tags: list[str],
) -> SummaryStats:
    """Аналог calculate_summary_stats из stage_two.py, но для тегов."""
    predictions = [m.result for m in metrics]
    durations   = [m.duration for m in metrics]
    tokens      = [m.tokens for m in metrics if m.tokens > 0]
    errors      = [m for m in metrics if m.error is not None]

    accuracy, successful = calc_morpho_accuracy(true_tags, predictions)

    return SummaryStats(
        total_queries=len(metrics),
        successful_predictions=successful,
        accuracy=accuracy,
        total_time=sum(durations),
        avg_time=sum(durations) / len(durations) if durations else 0,
        min_time=min(durations) if durations else 0,
        max_time=max(durations) if durations else 0,
        total_tokens=sum(tokens),
        avg_tokens=sum(tokens) / len(tokens) if tokens else 0,
        min_tokens=min(tokens) if tokens else 0,
        max_tokens=max(tokens) if tokens else 0,
        tokens_per_second=sum(tokens) / sum(durations)
            if durations and sum(durations) > 0 else 0,
        error_count=len(errors),
    )


# =============================================================================
# ИЗМЕНЕНИЕ 4: main (заменяет main из stage_two.py)
# =============================================================================

def parse_args():
    p = ArgumentParser()
    p.add_argument("--model",  default="claude-sonnet-4-6")
    p.add_argument("--corpus", required=True, help="Путь к CSV-файлу корпуса")
    p.add_argument("--tbox-path",             default="./tbox.json")
    p.add_argument("--tbox-interpreter-path", default="./interpreter.py")
    p.add_argument("--limit",  type=int, default=-1)
    p.add_argument("--debug",  action="store_true")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    # ── Загружаем корпус ──────────────────────────────────────────────────────
    data = load_udmurt_data(args.corpus)
    print(f"Загружено предложений: {len(data)}")

    metrics:   list[QueryMetrics] = []
    true_tags: list[str]          = []

    for _, row in tqdm(data.iterrows(), total=len(data)):
        if args.limit >= 0 and len(metrics) >= args.limit:
            break

        # Формируем task_data в том же стиле, что solar-режим в stage_two.py
        task_data = json.dumps({
            "sent_id":          row["sent_id"],
            "sentence":         row["sentence"],
            "uniparser_output": row["uniparser_output"],
        })

        from functools import partial
        predict_fn = partial(
            predict_solar_udmurt,
            tbox_path=args.tbox_path,
            tbox_interpreter_path=args.tbox_interpreter_path,
        )

        query_metrics = predict_with_metrics(
            predict_fn, args.model, task_data, args.debug, args.no_cache
        )

        metrics.append(query_metrics)
        true_tags.append(row.get("gold_tag", ""))

        if args.debug:
            print(f"  Эталон:    {row.get('gold_tag', '—')}")
            print(f"  Предсказ.: {query_metrics.result}")

    # ── Считаем метрики ───────────────────────────────────────────────────────
    stats = calculate_summary_stats_udmurt(metrics, true_tags)
    print_summary_stats(stats, len(data))

    # ── Сохраняем результаты ──────────────────────────────────────────────────
    out = [
        {
            "sent_id":   data.iloc[i]["sent_id"],
            "sentence":  data.iloc[i]["sentence"],
            "gold":      true_tags[i],
            "predicted": metrics[i].result,
            "correct":   true_tags[i] != "" and true_tags[i] in str(metrics[i].result or ""),
        }
        for i in range(len(metrics))
    ]
    Path("results_udmurt.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Результаты → results_udmurt.json")


if __name__ == "__main__":
    main()
