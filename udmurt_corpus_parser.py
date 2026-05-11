"""
udmurt_corpus_parser.py
=======================
Конвертер корпуса удмуртского языка (вывод UniParser) в формат ABox
для использования в пайплайне Solar.

Место в проекте: положить в корень проекта, рядом с prompts.py
"""

import re
import json
import sys
from pathlib import Path
from typing import Optional

# ─── Плейсхолдер для '||' ────────────────────────────────────────────────────
# Заменяем ' || ' перед split(), чтобы не разрезать внутри одного токена
_PIPE = "\x00"


# ─── 1. Разбивка строки на группы альтернатив ────────────────────────────────

def _split_to_groups(text: str) -> list[list[str]]:
    """
    Превращает строку вида  'революция кутск-и-з || кут-ск-и-з'
    в список групп:         [['революция'], ['кутск-и-з', 'кут-ск-и-з']]

    Пробел  = граница токенов
    ' || '  = граница альтернатив внутри одного токена
    """
    text = text.replace(" || ", _PIPE)
    groups = text.split()                          # разбиваем по реальным пробелам
    return [g.split(_PIPE) for g in groups]        # раскрываем альтернативы


# ─── 2. Маппинги тегов UniParser → UD-подобные признаки ─────────────────────

_TENSE    = {"PST": "Past", "PRS": "Present", "FUT": "Future"}
_NUMBER   = {"SG": "Singular", "PL": "Plural"}
_CASE     = {
    "ACC":  "Accusative",       "GEN":  "Genitive",
    "DAT":  "Dative",           "LOC":  "Locative",
    "ABL":  "Ablative",         "ILL":  "Illative",
    "EL":   "Elative",          "INS":  "Instrumental",
    "ADV":  "Adverbial",        "PROL": "Prolative",
    "DELIM":"Delimitative",
}
_VERBFORM = {"INF": "Infinitive", "CVB": "Converb",
             "VN":  "VerbalNoun", "PTCP": "Participle"}
_VOICE    = {"PASS": "Passive", "CAUS": "Causative", "ACT": "Active"}
_MOOD     = {"IMP": "Imperative", "DEB": "Debitive"}
_ASPECT   = {"ITER": "Iterative", "RES": "Resultative"}
_MISC     = {
    "EVID": "Evidential", "ORD":  "Ordinal",
    "COMP": "Comparative","ATTR": "Attributive",
}

# Ключи → предикаты ABox
_FEAT_TO_PRED = {
    "Tense":    "hasCandidateTense",
    "Number":   "hasCandidateNumber",
    "Person":   "hasCandidatePerson",
    "Case":     "hasCandidateCase",
    "Voice":    "hasCandidateVoice",
    "Mood":     "hasCandidateMood",
    "VerbForm": "hasCandidateVerbForm",
    "Aspect":   "hasCandidateAspect",
    "Polarity": "hasCandidatePolarity",
}


# ─── 3. Парсер одного тега ───────────────────────────────────────────────────

def _parse_tag(tag: str) -> dict:
    """
    Разбирает тег вида 'STEM-PASS-PST-3SG' в словарь признаков.
    Возвращает dict с ключами: Tense, Number, Person, Case, Voice, VerbForm, ...
    """
    parts = tag.split("-")
    if parts and parts[0] == "STEM":
        parts = parts[1:]           # убираем обязательный префикс STEM

    features: dict = {}

    for part in parts:
        if not part:
            continue

        # Составные теги вида PRS.12, FUT.3PL, PST.3SG ─────────────────────
        m_compound = re.match(r"^(PST|PRS|FUT)\.([123]{1,2})(SG|PL)?$", part)
        if m_compound:
            features["Tense"]  = _TENSE[m_compound.group(1)]
            features["Person"] = m_compound.group(2)
            if m_compound.group(3):
                features["Number"] = _NUMBER[m_compound.group(3)]
            continue

        # Посессивный маркер: P.3SG, P.1PL, P.3PL ...
        m_poss = re.match(r"^P\.([123]{1,2})(SG|PL)?$", part)
        if m_poss:
            features["PossPerson"] = m_poss.group(1)
            if m_poss.group(2):
                features["PossNumber"] = _NUMBER.get(m_poss.group(2), m_poss.group(2))
            continue

        # Личное окончание с числом: 3SG, 3PL, 1PL, 12, 3PL.ACC ...
        # (иногда Person и Number слиты)
        m_pn = re.match(r"^(12|[123])(SG|PL)?$", part)
        if m_pn:
            features["Person"] = m_pn.group(1)
            if m_pn.group(2):
                features["Number"] = _NUMBER[m_pn.group(2)]
            continue

        # Стандартные маппинги
        if part in _TENSE:
            features["Tense"] = _TENSE[part]
        elif part in _NUMBER:
            features["Number"] = _NUMBER[part]
        elif part in _CASE:
            features["Case"] = _CASE[part]
        elif part in _VERBFORM:
            features["VerbForm"] = _VERBFORM[part]
        elif part in _VOICE:
            features["Voice"] = _VOICE[part]
        elif part in _MOOD:
            features["Mood"] = _MOOD[part]
        elif part in _ASPECT:
            features["Aspect"] = _ASPECT[part]
        elif part in _MISC:
            features[part] = _MISC[part]
        elif part == "NEG":
            features["Polarity"] = "Negative"
        else:
            # Неизвестный тег — сохраняем как есть для отладки
            features[f"_raw_{part}"] = part

    return features


# ─── 4. Определение POS по признакам ─────────────────────────────────────────

def _infer_pos(features: dict, raw_tag: str) -> str:
    """Определяет вероятный POS по набору признаков тега."""
    vf = features.get("VerbForm", "")
    if vf == "VerbalNoun":
        return "Noun"
    if vf in ("Infinitive", "Converb", "Participle"):
        return "Verb"
    if "Tense" in features or "Person" in features or "Mood" in features:
        return "Verb"
    if "Case" in features or "PossPerson" in features:
        return "Noun"
    if raw_tag.strip() == "STEM":
        return "Noun"       # голый STEM = нарицательное / неизменяемое
    return "Unknown"


# ─── 5. Основная функция конвертации ─────────────────────────────────────────

def uniparser_to_abox(
    sentence: str,
    uniparser_output: str,
    sent_id: str = "sent1",
) -> dict:
    """
    Преобразует вывод UniParser в формат ABox для функции disambiguate().

    Args:
        sentence:         исходный текст предложения, напр. "Революция кутскиз."
        uniparser_output: строка из корпуса:
                          "революция кутск-и-з || кут-ск-и-з\tSTEM STEM-PST-3SG || STEM-PASS-PST-3SG"
                          (леммы + морф. разбивка TAB теги)
        sent_id:          идентификатор предложения

    Returns:
        {"individuals": {...}, "assertions": [...]}
    """
    uniparser_output = uniparser_output.strip()

    # ── Разбиваем по табуляции на леммы и теги ───────────────────────────────
    if "\t" in uniparser_output:
        lemma_line, tag_line = uniparser_output.split("\t", 1)
    else:
        # Только теги без лемм
        lemma_line = ""
        tag_line   = uniparser_output

    tag_groups   = _split_to_groups(tag_line)
    lemma_groups = _split_to_groups(lemma_line) if lemma_line else [
        [tok] for tok in tag_line.split()
    ]

    # ── Поверхностные токены из исходного текста ─────────────────────────────
    surface_tokens = [
        t for t in re.split(r"\s+", sentence.strip())
        if re.search(r"\w", t, re.UNICODE)
    ]

    n = min(len(lemma_groups), len(tag_groups), len(surface_tokens))

    # ── Инициализируем ABox ───────────────────────────────────────────────────
    individuals: dict = {sent_id: {"type": "Sentence"}}
    assertions:  list = [
        {"predicate": "hasText", "args": [sent_id, sentence]}
    ]

    # ── Строим индивиды и ассерции ────────────────────────────────────────────
    for tok_idx in range(n):
        tok_id  = f"tok{tok_idx + 1}"
        surface = re.sub(r"[.,;:!?«»—\n]", "", surface_tokens[tok_idx])

        individuals[tok_id] = {"type": "Token"}
        assertions += [
            {"predicate": "belongsToSentence", "args": [tok_id, sent_id]},
            {"predicate": "hasForm",           "args": [tok_id, surface]},
            {"predicate": "hasPosition",       "args": [tok_id, str(tok_idx + 1)]},
        ]

        lemma_alts = lemma_groups[tok_idx]
        tag_alts   = tag_groups[tok_idx]
        n_alts     = min(len(lemma_alts), len(tag_alts))

        for alt_idx in range(n_alts):
            ana_id    = f"ana_tok{tok_idx + 1}_alt{alt_idx + 1}"
            lemma_raw = lemma_alts[alt_idx].strip()
            tag_raw   = tag_alts[alt_idx].strip()

            # Лемма = первая морфема до дефиса (грубое приближение)
            lemma_clean = lemma_raw.split("-")[0] if "-" in lemma_raw else lemma_raw

            features = _parse_tag(tag_raw)
            pos      = _infer_pos(features, tag_raw)

            individuals[ana_id] = {"type": "CandidateAnalysis"}
            assertions += [
                {"predicate": "candidateOf",       "args": [ana_id, tok_id]},
                {"predicate": "hasLemma",          "args": [ana_id, lemma_clean]},
                {"predicate": "hasMorphBreakdown", "args": [ana_id, lemma_raw]},
                {"predicate": "hasRawTag",         "args": [ana_id, tag_raw]},
            ]

            if pos != "Unknown":
                assertions.append(
                    {"predicate": "hasCandidatePOS", "args": [ana_id, pos]}
                )

            # Добавляем каждый признак отдельной ассерцией
            for feat_key, predicate in _FEAT_TO_PRED.items():
                if feat_key in features:
                    assertions.append(
                        {"predicate": predicate, "args": [ana_id, features[feat_key]]}
                    )

    return {"individuals": individuals, "assertions": assertions}


# ─── 6. Парсер корпусного файла ───────────────────────────────────────────────

def parse_corpus_file(filepath: str) -> list[dict]:
    """
    Читает CSV-файл корпуса (разделитель ;) и возвращает список записей:
    [{"sent_id", "sentence", "uniparser_output", "source"}, ...]
    """
    results = []
    path    = Path(filepath)

    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(";")
            if len(parts) < 3:
                continue

            source = parts[0].strip('"').strip()

            # Ищем колонку с текстом (содержит {{ ... }})
            sentence_col  = None
            analysis_col  = None

            for i, col in enumerate(parts):
                if "{{" in col:
                    sentence_col = i
                    # Следующая непустая колонка — анализ UniParser
                    for j in range(i + 1, len(parts)):
                        if parts[j].strip():
                            analysis_col = j
                            break
                    break

            if sentence_col is None:
                continue

            raw_sentence = parts[sentence_col].strip()
            sentence     = re.sub(r"[{}]", "", raw_sentence).strip()

            uniparser_output = ""
            if analysis_col is not None:
                uniparser_output = parts[analysis_col].strip().strip('"')

            if not sentence or not uniparser_output:
                continue

            results.append({
                "sent_id":          f"sent_{line_no}",
                "sentence":         sentence,
                "uniparser_output": uniparser_output,
                "source":           source,
            })

    return results


def corpus_to_aboxes(filepath: str) -> list[dict]:
    """Конвертирует весь корпусный файл в список ABox-словарей."""
    records = parse_corpus_file(filepath)
    return [
        {
            "sent_id":  r["sent_id"],
            "sentence": r["sentence"],
            "source":   r["source"],
            "abox":     uniparser_to_abox(
                            r["sentence"],
                            r["uniparser_output"],
                            r["sent_id"],
                        ),
        }
        for r in records
    ]


# ─── 7. Точка входа (CLI) ─────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Быстрая проверка на двух примерах из корпуса ──────────────────────
    examples = [
        (
            "Революция кутскиз.",
            "революция кутск-и-з || кут-ск-и-з\t"
            "STEM STEM-PST-3SG || STEM-PASS-PST-3SG",
        ),
        (
            "Уллясько, нош со уг но вырӟы.",
            "улля-сько || улля-ськ-о || улля-ськ-о нош со уг но вырӟ-ы\t"
            "STEM-PRS.12 || STEM-PASS-PRS.3PL || STEM-PASS-FUT STEM STEM STEM STEM STEM-NEG.SG",
        ),
    ]

    for sentence, uni_out in examples:
        print("=" * 60)
        print(f"Предложение: {sentence}")
        abox = uniparser_to_abox(sentence, uni_out)
        print(json.dumps(abox, ensure_ascii=False, indent=2))

    # ── Если передан путь к файлу корпуса — конвертируем целиком ──────────
    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
        print(f"\n=== Конвертируем: {corpus_path} ===")
        aboxes  = corpus_to_aboxes(corpus_path)
        out     = corpus_path.replace(".csv", "_aboxes.json")
        Path(out).write_text(
            json.dumps(aboxes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Готово: {len(aboxes)} предложений → {out}")
