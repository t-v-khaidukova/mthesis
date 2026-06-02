#!/usr/bin/env python3
"""
stage_udmurt_two_sourcegrounded.py
===================================

Самостоятельный второй этап для удмуртского корпуса, использующий source-grounded parser с автоматическим анализом поля
cands_analtag_line. Скрипт:

1) запускает текущий interpreter.py через disambiguate();
2) сравнивает результат интерпретатора с gold_tag_line;
3) отдельно анализирует исходные кандидаты UniParser из cands_analtag_line:
   - сколько токенов однозначны/неоднозначны;
   - есть ли gold-тег среди кандидатов;
   - сколько неоднозначных случаев потенциально можно было разрешать.

Важно: baseline top-1 здесь НЕ считается, потому что UniParser в cands_analtag_line
не обязательно выбирает один итоговый вариант, а хранит набор кандидатов.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from udmurt_corpus_parser_sourcegrounded import uniparser_to_abox, explain_disambiguation


def safe_filename_part(value: str) -> str:
    """
    Делает строку безопасной для имени файла.

    Пример:
        "openai/gpt-5-mini" -> "openai_gpt-5-mini"
    """
    value = (value or "model").strip()
    value = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_.-]+", "_", value)
    value = value.strip("._-")
    return value or "model"


def build_output_paths(args) -> tuple[Path, Path, Path]:
    """
    Строит имена выходных файлов.

    Если --output / --summary-output / --token-output не заданы явно,
    файлы сохраняются с именем модели:
        results_udmurt_<model>.json
        results_udmurt_<model>_summary.json
        results_udmurt_<model>_token_candidates.csv
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = safe_filename_part(args.model)

    output_path = (
        Path(args.output)
        if args.output
        else output_dir / f"results_udmurt_{model_name}.json"
    )
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else output_dir / f"results_udmurt_{model_name}_summary.json"
    )
    token_output_path = (
        Path(args.token_output)
        if args.token_output
        else output_dir / f"results_udmurt_{model_name}_token_candidates.csv"
    )

    return output_path, summary_path, token_output_path



# =============================================================================
# 0. Простые структуры для измерения выполнения
# =============================================================================

@dataclass
class QueryMetrics:
    result: str | None
    duration: float
    tokens: int = 0
    error: str | None = None


def predict_with_metrics(
    prediction_func,
    model_name: str,
    task_data: str,
    debug: bool,
    no_cache: bool,
) -> QueryMetrics:
    """
    Локальная версия predict_with_metrics без зависимости от stage_two.py.
    Здесь LLM не вызывается: model_name/no_cache оставлены только для совместимости CLI.
    """
    start_time = time.time()
    result: str | None = None
    error: str | None = None

    try:
        result = prediction_func(model_name, task_data, debug, no_cache)
    except Exception as ex:  # noqa: BLE001
        error = str(ex)
        if debug:
            print(f"Error in prediction: {ex}")
            traceback.print_exc()

    return QueryMetrics(
        result=result,
        duration=time.time() - start_time,
        tokens=0,
        error=error,
    )


# =============================================================================
# 1. Загрузка и кеширование interpreter.py
# =============================================================================

@cache
def _load_interpreter(tbox_interpreter_path: str) -> dict[str, Any]:
    """
    Один раз компилирует interpreter.py и возвращает namespace.
    Кешируется по пути файла.
    """
    code = Path(tbox_interpreter_path).read_text(encoding="utf-8")
    ns: dict[str, Any] = {}
    exec(compile(code, tbox_interpreter_path, "exec"), ns)
    if "disambiguate" not in ns:
        if "calculate" in ns:
            raise RuntimeError(
                f"В файле {tbox_interpreter_path} найдена функция calculate(), "
                "но для удмуртского этапа нужна функция disambiguate(abox, sentence_id). "
                "Скорее всего, это старый SARA/legal-интерпретатор, а не языковой interpreter.py."
            )
        raise RuntimeError(
            f"В файле {tbox_interpreter_path} не найдена функция "
            "disambiguate(abox, sentence_id)."
        )
    return ns


def get_disambiguate(tbox_interpreter_path: str):
    return _load_interpreter(tbox_interpreter_path)["disambiguate"]


# =============================================================================
# 2. Загрузка данных
# =============================================================================

def _is_valid_sentence(sentence: str) -> bool:
    """Фильтрует мусор: числа, пустые строки, строки без кириллицы."""
    s = sentence.strip()
    if len(s) < 3:
        return False
    if not re.search(r"[а-яёА-ЯЁӝӟӥӧӵ]", s, re.UNICODE):
        return False
    return True


def load_udmurt_data(corpus_path: str) -> pd.DataFrame:
    """
    Читает датасет с колонками:
        tokens_braced      — предложение с {{...}} вокруг токенов;
        cands_analtag_line — строка UniParser: морфемный/лемматизированный разбор + tab + теги;
        gold_analysis_line — эталонный морфологический анализ;
        gold_tag_line      — эталонные теги для оценки.
    """
    df_raw = pd.read_csv(corpus_path, sep=";", dtype=str, keep_default_na=False)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    required = {"tokens_braced", "cands_analtag_line", "gold_tag_line"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"Не найдены колонки: {missing}\n"
            f"Доступные колонки: {list(df_raw.columns)}"
        )

    rows: list[dict[str, str]] = []
    for idx, row in df_raw.iterrows():
        sentence_braced = str(row.get("tokens_braced", "")).strip()
        uni_out = str(row.get("cands_analtag_line", "")).strip()
        gold_tag = str(row.get("gold_tag_line", "")).strip()
        gold_analysis = str(row.get("gold_analysis_line", "")).strip()

        sentence = re.sub(r"[{}]", "", sentence_braced).strip()
        sentence = re.sub(r"\s+", " ", sentence)

        if not _is_valid_sentence(sentence):
            print(f"  ОТФИЛЬТРОВАНО [{idx}]: {repr(sentence[:60])}")
            continue
        if not uni_out:
            continue

        rows.append({
            "sent_id": f"sent_{idx}",
            "sentence": sentence,
            "uniparser_output": uni_out,
            "gold_tag": gold_tag,
            "gold_analysis": gold_analysis,
            "source": str(row.get("doc_id", "")),
        })

    df = pd.DataFrame(rows)
    print(f"  Загружено: {len(df)} / {len(df_raw)} записей (после фильтрации)")
    return df


# =============================================================================
# 3. Разбор cands_analtag_line и анализ неоднозначности UniParser
# =============================================================================

def extract_tag_line(cands_analtag_line: str) -> str:
    """
    Извлекает строку морфологических тегов из поля cands_analtag_line.

    Формат поля обычно такой:
        morph_or_lemma_line \t tag_line

    Например:
        котьку мед пишт-о-з\tSTEM STEM STEM-FUT-3SG

    Сравнивать с gold_tag_line нужно только tag_line, а не левую часть.
    """
    text = (cands_analtag_line or "").strip()

    # Нормальный случай: внутри ячейки есть настоящая табуляция.
    if "\t" in text:
        return text.rsplit("\t", 1)[1].strip()

    # На случай, если табуляция была сохранена как два символа: \ + t.
    if "\\t" in text:
        return text.rsplit("\\t", 1)[1].strip()

    # Если на вход уже передали только строку тегов.
    return text


def split_candidate_tag_groups(tag_line: str) -> list[list[str]]:
    """
    Делит строку тегов на группы кандидатов по токенам.

    Пример:
        STEM || STEM-EGR STEM-PASS-FUT-COMP || STEM-PRS.12-COMP STEM

    Результат:
        [
            ["STEM", "STEM-EGR"],
            ["STEM-PASS-FUT-COMP", "STEM-PRS.12-COMP"],
            ["STEM"],
        ]

    Дубликаты сохраняются, потому что они могут соответствовать разным морфемным
    разбором при одинаковом теге. Для оценки tag-level неоднозначности ниже
    дополнительно используется множество unique_candidates.
    """
    text = (tag_line or "").strip()
    if not text:
        return []

    # Все пробелы вокруг || заменяем временным символом, чтобы альтернативы
    # остались внутри одной группы токена при последующем split().
    text = re.sub(r"\s*\|\|\s*", "\x00", text)
    groups = text.split()
    return [group.split("\x00") for group in groups if group]


def analyze_uniparser_candidates_for_sentence(
    sentence: str,
    gold_tag_line: str,
    cands_analtag_line: str,
    sent_id: str = "",
) -> dict[str, Any]:
    """
    Анализирует кандидаты UniParser для одного предложения.

    Считает два типа неоднозначности:
      1) raw_analysis_ambiguous: есть несколько альтернатив через ||;
      2) tag_ambiguous: среди альтернатив есть больше одного различного тега.

    Для сравнения с gold_tag_line важнее tag_ambiguous, потому что gold_tag_line
    содержит именно теги, а не морфемные варианты.
    """
    gold_tokens = (gold_tag_line or "").strip().split()
    tag_line = extract_tag_line(cands_analtag_line)
    candidate_groups = split_candidate_tag_groups(tag_line)
    surface_tokens = sentence.strip().split()

    length_mismatch = len(gold_tokens) != len(candidate_groups)
    n = min(len(gold_tokens), len(candidate_groups))

    token_rows: list[dict[str, Any]] = []

    totals = {
        "compared_tokens": 0,
        "raw_analysis_ambiguous_tokens": 0,
        "tag_ambiguous_tokens": 0,
        "tag_unambiguous_tokens": 0,
        "gold_in_candidates": 0,
        "gold_not_in_candidates": 0,
        "unambiguous_correct": 0,
        "unambiguous_incorrect": 0,
        "tag_ambiguous_gold_in_candidates": 0,
        "tag_ambiguous_gold_not_in_candidates": 0,
    }

    for i in range(n):
        gold = gold_tokens[i]
        candidates_raw = candidate_groups[i]
        unique_candidates = list(dict.fromkeys(candidates_raw))

        raw_analysis_ambiguous = len(candidates_raw) > 1
        tag_ambiguous = len(unique_candidates) > 1
        gold_present = gold in unique_candidates

        totals["compared_tokens"] += 1
        if raw_analysis_ambiguous:
            totals["raw_analysis_ambiguous_tokens"] += 1
        if tag_ambiguous:
            totals["tag_ambiguous_tokens"] += 1
            if gold_present:
                totals["tag_ambiguous_gold_in_candidates"] += 1
            else:
                totals["tag_ambiguous_gold_not_in_candidates"] += 1
        else:
            totals["tag_unambiguous_tokens"] += 1
            if gold_present:
                totals["unambiguous_correct"] += 1
            else:
                totals["unambiguous_incorrect"] += 1

        if gold_present:
            totals["gold_in_candidates"] += 1
        else:
            totals["gold_not_in_candidates"] += 1

        token_rows.append({
            "sent_id": sent_id,
            "token_index": i,
            "surface": surface_tokens[i] if i < len(surface_tokens) else "",
            "gold_tag": gold,
            "candidate_tags_raw": candidates_raw,
            "candidate_tags_unique": unique_candidates,
            "raw_candidate_count": len(candidates_raw),
            "unique_tag_count": len(unique_candidates),
            "raw_analysis_ambiguous": raw_analysis_ambiguous,
            "tag_ambiguous": tag_ambiguous,
            "gold_in_candidates": gold_present,
        })

    return {
        "sent_id": sent_id,
        "sentence": sentence,
        "gold_token_count": len(gold_tokens),
        "candidate_token_count": len(candidate_groups),
        "length_mismatch": length_mismatch,
        "tag_line": tag_line,
        "token_rows": token_rows,
        **totals,
    }


def calc_uniparser_candidate_metrics(sentence_analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """Сводит анализ кандидатов UniParser по всем предложениям."""
    keys = [
        "compared_tokens",
        "raw_analysis_ambiguous_tokens",
        "tag_ambiguous_tokens",
        "tag_unambiguous_tokens",
        "gold_in_candidates",
        "gold_not_in_candidates",
        "unambiguous_correct",
        "unambiguous_incorrect",
        "tag_ambiguous_gold_in_candidates",
        "tag_ambiguous_gold_not_in_candidates",
    ]

    out: dict[str, Any] = {k: 0 for k in keys}
    out["sentences"] = len(sentence_analyses)
    out["length_mismatch_sentences"] = 0

    for analysis in sentence_analyses:
        for k in keys:
            out[k] += int(analysis.get(k, 0))
        if analysis.get("length_mismatch"):
            out["length_mismatch_sentences"] += 1

    total = out["compared_tokens"]
    out["raw_analysis_ambiguous_rate"] = (
        out["raw_analysis_ambiguous_tokens"] / total if total else 0
    )
    out["tag_ambiguous_rate"] = out["tag_ambiguous_tokens"] / total if total else 0
    out["gold_in_candidates_rate"] = out["gold_in_candidates"] / total if total else 0
    out["unambiguous_correct_rate"] = (
        out["unambiguous_correct"] / out["tag_unambiguous_tokens"]
        if out["tag_unambiguous_tokens"] else 0
    )
    return out


# =============================================================================
# 4. Детерминированная разметка через interpreter.py
# =============================================================================

def _get_raw_tags(abox: dict) -> dict[str, str]:
    """candidate_id → исходный UniParser-тег из hasRawTag."""
    raw_tags: dict[str, str] = {}
    for a in abox.get("assertions", []):
        if a.get("predicate") == "hasRawTag" and len(a.get("args", [])) == 2:
            raw_tags[a["args"][0]] = a["args"][1]
    return raw_tags


def _count_candidates(abox: dict) -> dict[str, int]:
    """token_id → количество кандидатов."""
    counts: dict[str, int] = {}
    for a in abox.get("assertions", []):
        if a.get("predicate") == "candidateOf":
            tok = a["args"][1]
            counts[tok] = counts.get(tok, 0) + 1
    return counts


def _format_result(disamb_result: dict, abox: dict) -> str:
    """
    Форматирует результат в строку тегов, совместимую с gold_tag_line.

    Пример:
        STEM STEM-PASS-PRS.3SG STEM-CVB

    Неразрешённые токены:
        UNRESOLVED[STEM-PST-3SG||STEM-PASS-PST-3SG]
    """
    raw_tags = _get_raw_tags(abox)
    parts: list[str] = []

    for tok_id, info in disamb_result.get("tokens", {}).items():
        status = info.get("status", "unknown")
        if status == "resolved":
            selected = info.get("selected_candidate")
            parts.append(raw_tags.get(selected, selected or "?"))
        elif status == "unresolved":
            cands = info.get("remaining_candidates", [])
            tags = [raw_tags.get(c, c) for c in cands]
            parts.append(f"UNRESOLVED[{'||'.join(tags)}]")
        elif status == "contradiction":
            parts.append("CONTRADICTION")
        elif status == "no_candidates":
            parts.append("NO_CANDS")
        else:
            parts.append(str(status).upper())

    return " ".join(parts) if parts else "no_tokens"


def predict_solar_udmurt(
    model: str,
    task: str,
    debug: bool,
    no_cache: bool,
    tbox_interpreter_path: str,
    explanations_store: dict[str, list[dict[str, Any]]] | None = None,
    **_kwargs,
) -> str | None:
    """
    Детерминированная разметка:
      1. Парсим UniParser-вывод → ABox;
      2. Вызываем disambiguate() из interpreter.py напрямую;
      3. Сохраняем объяснения;
      4. Форматируем результат в строку тегов.
    """
    try:
        task_data = json.loads(task)
    except json.JSONDecodeError as e:
        if debug:
            print(f"  [!] Ошибка парсинга task JSON: {e}")
        return None

    sent_id = task_data["sent_id"]
    sentence = task_data["sentence"]
    uniparser_output = task_data["uniparser_output"]

    try:
        abox = uniparser_to_abox(sentence, uniparser_output, sent_id)
    except Exception as e:  # noqa: BLE001
        if debug:
            print(f"  [!] uniparser_to_abox failed: {e}")
            traceback.print_exc()
        return None

    if debug:
        cand_counts = _count_candidates(abox)
        n_ambiguous = sum(1 for v in cand_counts.values() if v > 1)
        print(f"\n[{sent_id}] {sentence[:80]}")
        print(f"  Токенов в ABox: {len(cand_counts)}  Неоднозначных по ABox: {n_ambiguous}")

    try:
        disambiguate_fn = get_disambiguate(tbox_interpreter_path)
        result = disambiguate_fn(abox, sent_id)
    except Exception as e:  # noqa: BLE001
        if debug:
            print(f"  [!] disambiguate() failed: {e}")
            traceback.print_exc()
        return None

    token_explanations = explain_disambiguation(result, abox)
    if explanations_store is not None:
        explanations_store[sent_id] = token_explanations

    if debug:
        for e in token_explanations:
            if e.get("was_ambiguous"):
                print(f"  [{e.get('surface')}] {e.get('status')} "
                      f"→ {e.get('selected_tag') or 'UNRESOLVED'}")
                if e.get("reason"):
                    print(f"    Причина: {e.get('reason')}")

    formatted = _format_result(result, abox)

    if debug:
        print(f"  Результат: {formatted}")

    return formatted


# =============================================================================
# 5. Метрики результата интерпретатора
# =============================================================================

def calc_morpho_metrics(
    true_tags: list[str],
    predictions: list[str | None],
) -> dict[str, Any]:
    """
    Token-level метрики для результата интерпретатора:
      - resolution_rate: доля токенов, получивших однозначный ответ;
      - precision_on_resolved: точность на разрешённых токенах;
      - accuracy_on_all_tokens: правильные разрешённые / все gold-токены.
    """
    total = 0
    resolved = 0
    resolved_correct = 0
    unresolved = 0
    unresolved_with_gold = 0
    missing_prediction = 0
    extra_prediction_tokens = 0
    length_mismatch_sentences = 0

    for gold, pred in zip(true_tags, predictions):
        if not gold:
            continue

        gold_toks = gold.strip().split()
        pred_toks = pred.strip().split() if pred else []

        if len(gold_toks) != len(pred_toks):
            length_mismatch_sentences += 1
            if len(pred_toks) > len(gold_toks):
                extra_prediction_tokens += len(pred_toks) - len(gold_toks)

        for i, gold_tag in enumerate(gold_toks):
            total += 1

            if i >= len(pred_toks):
                missing_prediction += 1
                continue

            pred_tok = pred_toks[i]
            if pred_tok.startswith("UNRESOLVED"):
                unresolved += 1
                if gold_tag in pred_tok:
                    unresolved_with_gold += 1
                continue

            resolved += 1
            if pred_tok == gold_tag:
                resolved_correct += 1

    return {
        "total_tokens": total,
        "resolved": resolved,
        "resolved_correct": resolved_correct,
        "unresolved": unresolved,
        "unresolved_with_gold": unresolved_with_gold,
        "missing_prediction": missing_prediction,
        "extra_prediction_tokens": extra_prediction_tokens,
        "length_mismatch_sentences": length_mismatch_sentences,
        "resolution_rate": resolved / total if total else 0,
        "precision_on_resolved": resolved_correct / resolved if resolved else 0,
        "accuracy_on_all_tokens": resolved_correct / total if total else 0,
    }


def print_timing_stats(metrics: list[QueryMetrics]) -> None:
    durations = [m.duration for m in metrics]
    errors = [m for m in metrics if m.error is not None]

    print("\n--- Timing Statistics ---")
    print(f"Total queries processed: {len(metrics)}")
    print(f"Total time: {sum(durations):.3f}s")
    print(f"Average time per query: {(sum(durations) / len(durations)) if durations else 0:.3f}s")
    print(f"Min time per query: {min(durations) if durations else 0:.3f}s")
    print(f"Max time per query: {max(durations) if durations else 0:.3f}s")
    print(f"Errors: {len(errors)}")


# =============================================================================
# 6. CLI
# =============================================================================

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini",
                        help=(
                            "Имя модели/режима для подписи результатов. "
                            "На втором этапе LLM не вызывается: используется готовый interpreter.py."
                        ))
    parser.add_argument("--corpus", required=True,
                        help="Путь к CSV-файлу корпуса")
    parser.add_argument("--tbox-path", default="./tbox.json",
                        help="Путь к TBox JSON; оставлено для совместимости, напрямую не используется")
    parser.add_argument("--tbox-interpreter-path", default="./interpreter.py",
                        help="Путь к interpreter.py с функцией disambiguate()")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Максимальное число предложений (-1 = все)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output-dir", default=".",
                        help="Папка для выходных файлов")
    parser.add_argument("--output", default=None,
                        help=(
                            "JSON с результатами по предложениям. "
                            "Если не задано, имя строится автоматически по --model."
                        ))
    parser.add_argument("--summary-output", default=None,
                        help=(
                            "JSON со сводными метриками. "
                            "Если не задано, имя строится автоматически по --model."
                        ))
    parser.add_argument("--token-output", default=None,
                        help=(
                            "CSV с анализом кандидатов по токенам. "
                            "Если не задано, имя строится автоматически по --model."
                        ))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, summary_path, token_output_path = build_output_paths(args)

    print(f"Загружаем корпус: {args.corpus}")
    data = load_udmurt_data(args.corpus)
    print(f"Предложений для обработки: {len(data)}")

    if len(data) == 0:
        print("Нет валидных предложений. Проверьте формат CSV и фильтры.")
        return

    metrics: list[QueryMetrics] = []
    true_tags: list[str] = []
    explanations_store: dict[str, list[dict[str, Any]]] = {}

    sentence_candidate_analyses: list[dict[str, Any]] = []
    all_token_candidate_rows: list[dict[str, Any]] = []

    max_items = min(len(data), args.limit if args.limit >= 0 else len(data))

    for _, row in tqdm(data.iterrows(), total=max_items):
        if args.limit >= 0 and len(metrics) >= args.limit:
            break

        sent_id = row["sent_id"]
        sentence = row["sentence"]
        gold_tag = row.get("gold_tag", "")
        uniparser_output = row["uniparser_output"]

        # Анализ кандидатов UniParser до запуска интерпретатора.
        cand_analysis = analyze_uniparser_candidates_for_sentence(
            sentence=sentence,
            gold_tag_line=gold_tag,
            cands_analtag_line=uniparser_output,
            sent_id=sent_id,
        )
        sentence_candidate_analyses.append(cand_analysis)
        all_token_candidate_rows.extend(cand_analysis["token_rows"])

        task_data = json.dumps({
            "sent_id": sent_id,
            "sentence": sentence,
            "uniparser_output": uniparser_output,
        }, ensure_ascii=False)

        predict_fn = partial(
            predict_solar_udmurt,
            tbox_interpreter_path=args.tbox_interpreter_path,
            explanations_store=explanations_store,
        )

        query_metrics = predict_with_metrics(
            predict_fn,
            args.model,
            task_data,
            args.debug,
            args.no_cache,
        )

        metrics.append(query_metrics)
        true_tags.append(gold_tag)

        if args.debug:
            print(f"  Эталон:    {gold_tag or '—'}")
            print(f"  Предсказ.: {query_metrics.result}")

    # ── Метрики интерпретатора ───────────────────────────────────────────────
    morph = calc_morpho_metrics(true_tags, [m.result for m in metrics])
    print("\n── Морфологические метрики интерпретатора ───────────────")
    print(f"  Всего токенов в gold:              {morph['total_tokens']}")
    print(f"  Разрешено однозначно:              {morph['resolved']} "
          f"({morph['resolution_rate']:.1%})")
    print(f"  Правильных среди разрешённых:      {morph['resolved_correct']} / {morph['resolved']} "
          f"({morph['precision_on_resolved']:.1%})")
    print(f"  Правильных от всех токенов:        {morph['resolved_correct']} / {morph['total_tokens']} "
          f"({morph['accuracy_on_all_tokens']:.1%})")
    print(f"  Осталось UNRESOLVED:               {morph['unresolved']}")
    print(f"  UNRESOLVED, где gold был внутри:   {morph['unresolved_with_gold']}")
    print(f"  Нет предсказанного токена:         {morph['missing_prediction']}")
    print(f"  Предложений с разной длиной:       {morph['length_mismatch_sentences']}")

    # ── Метрики кандидатов UniParser ─────────────────────────────────────────
    uni = calc_uniparser_candidate_metrics(sentence_candidate_analyses)
    print("\n── Неоднозначность кандидатов UniParser ─────────────────")
    print(f"  Всего сопоставленных токенов:            {uni['compared_tokens']}")
    print(f"  Токены с несколькими raw-вариантами:     {uni['raw_analysis_ambiguous_tokens']} "
          f"({uni['raw_analysis_ambiguous_rate']:.1%})")
    print(f"  Токены с несколькими разными тегами:     {uni['tag_ambiguous_tokens']} "
          f"({uni['tag_ambiguous_rate']:.1%})")
    print(f"  Токены с одним тегом:                    {uni['tag_unambiguous_tokens']}")
    print(f"  Gold среди кандидатов UniParser:         {uni['gold_in_candidates']} "
          f"({uni['gold_in_candidates_rate']:.1%})")
    print(f"  Gold отсутствует среди кандидатов:       {uni['gold_not_in_candidates']}")
    print(f"  Однозначные по тегу совпали с gold:      {uni['unambiguous_correct']}")
    print(f"  Однозначные по тегу НЕ совпали с gold:   {uni['unambiguous_incorrect']}")
    print(f"  Неоднозначные по тегу, gold есть:        {uni['tag_ambiguous_gold_in_candidates']}")
    print(f"  Неоднозначные по тегу, gold отсутствует: {uni['tag_ambiguous_gold_not_in_candidates']}")
    print(f"  Предложений с несовпадением длины:       {uni['length_mismatch_sentences']}")

    print_timing_stats(metrics)

    # ── Сохраняем результаты по предложениям ────────────────────────────────
    rows_out: list[dict[str, Any]] = []
    for i in range(len(metrics)):
        pred = metrics[i].result or ""
        gold = true_tags[i]
        sent_id = data.iloc[i]["sent_id"]

        gold_toks = gold.split()
        pred_toks = pred.split()
        exact_sentence_match = (
            gold != ""
            and len(gold_toks) == len(pred_toks)
            and all(g == p for g, p in zip(gold_toks, pred_toks))
        )

        disambiguations = [
            {
                "surface": e.get("surface"),
                "status": e.get("status"),
                "selected_tag": e.get("selected_tag"),
                "rejected_tags": e.get("rejected_tags"),
                "all_candidates": e.get("all_candidates"),
                "reason": e.get("reason"),
            }
            for e in explanations_store.get(sent_id, [])
            if e.get("was_ambiguous")
        ]

        cand_analysis = sentence_candidate_analyses[i]
        rows_out.append({
            "sent_id": sent_id,
            "sentence": data.iloc[i]["sentence"],
            "gold_tags": gold,
            "gold_analysis": data.iloc[i].get("gold_analysis", ""),
            "uniparser_tag_line": cand_analysis.get("tag_line", ""),
            "uniparser_candidate_summary": {
                "compared_tokens": cand_analysis["compared_tokens"],
                "raw_analysis_ambiguous_tokens": cand_analysis["raw_analysis_ambiguous_tokens"],
                "tag_ambiguous_tokens": cand_analysis["tag_ambiguous_tokens"],
                "gold_in_candidates": cand_analysis["gold_in_candidates"],
                "gold_not_in_candidates": cand_analysis["gold_not_in_candidates"],
                "length_mismatch": cand_analysis["length_mismatch"],
            },
            "predicted": pred,
            "exact_sentence_match": exact_sentence_match,
            "error": metrics[i].error,
            "disambiguations": disambiguations,
        })

    output_path.write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "model": args.model,
        "tbox_path": args.tbox_path,
        "tbox_interpreter_path": args.tbox_interpreter_path,
        "interpreter_metrics": morph,
        "uniparser_candidate_metrics": uni,
        "processed_sentences": len(metrics),
        "loaded_sentences_after_filtering": len(data),
        "limit": args.limit,
        "output": str(output_path),
        "summary_output": str(summary_path),
        "token_output": str(token_output_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if all_token_candidate_rows:
        token_df = pd.DataFrame(all_token_candidate_rows)
        # Списки в CSV сохраняем как JSON-строки, чтобы их было удобно читать.
        for col in ["candidate_tags_raw", "candidate_tags_unique"]:
            token_df[col] = token_df[col].apply(lambda x: json.dumps(x, ensure_ascii=False))
        token_df.to_csv(token_output_path, sep=";", index=False)
    else:
        token_output_path.write_text("", encoding="utf-8")

    print(f"\nРезультаты по предложениям → {output_path}")
    print(f"Сводные метрики → {summary_path}")
    print(f"Кандидаты по токенам → {token_output_path}")


if __name__ == "__main__":
    main()
