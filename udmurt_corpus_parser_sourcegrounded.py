"""
udmurt_corpus_parser_sourcegrounded.py
======================================

Source-grounded converter from UniParser Udmurt output to the ABox format used
by stage_udmurt_two.py. Compared with the earlier version, this parser:

1) preserves every raw UniParser tag and every tag part via hasRawTagPart;
2) exposes all confidently parsed features, but does not hide unknown tag parts;
3) adds possessive person/number properties when tags contain P.1SG etc.;
4) tokenizes initials and abbreviations more robustly;
5) can create a conservative STEM candidate for initials/acronyms when the
   UniParser tag line has fewer groups than the surface tokenization.

The linguistic tags still come from UniParser data; the fallback STEM is used
only for obvious initials/acronyms/proper-name abbreviations that otherwise
produce missing predictions.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_PIPE = "\x00"


# ─── 1. Splitting and tokenization ────────────────────────────────────────────

def _split_to_groups(text: str) -> list[list[str]]:
    """Split a UniParser line into token groups, preserving alternatives separated by ||."""
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"\s*\|\|\s*", _PIPE, text)
    return [g.split(_PIPE) for g in text.split() if g]


def _clean_surface_token(tok: str) -> str:
    tok = (tok or "").strip()
    tok = tok.strip('"«»“”„()[]{}<>')
    # Remove punctuation, brackets, emoji and other non-word symbols around the token,
    # while keeping internal hyphens and letters/digits.
    tok = re.sub(r"^[^0-9A-Za-zА-Яа-яЁёӝӟӥӧӵӞӜӴӦ]+", "", tok)
    tok = re.sub(r"[^0-9A-Za-zА-Яа-яЁёӝӟӥӧӵӞӜӴӦ]+$", "", tok)
    return tok.strip()


def _tokenize_surface(sentence: str) -> list[str]:
    """
    Tokenize the visible sentence for ABox surface forms.

    This is not meant to replace UniParser tokenization; it is only for surface
    diagnostics/context. It expands initials such as "Э.С." into ["Э", "С"],
    which reduces missing tokens for social-media data with names/abbreviations.
    """
    s = re.sub(r"[{}]", "", sentence or "")
    s = s.replace("\\n", " ")
    s = re.sub(r"\s+", " ", s).strip()

    tokens: list[str] = []
    for raw0 in re.split(r"\s+", s):
        raw0 = (raw0 or "").strip()
        if not raw0:
            continue

        # Expand compact initials before stripping dots: Э.С., M.V., А.Г. -> Э, С / M, V / А, Г
        raw_for_initials = raw0.strip('"«»“”„()[]{}<>')
        compact_initials = re.findall(r"[A-ZА-ЯЁӞӜӴӦ]\.", raw_for_initials)
        compact_tail = re.sub(r"[A-ZА-ЯЁӞӜӴӦ]\.", "", raw_for_initials)
        compact_tail = re.sub(r"[,.;:!?—–\-]+", "", compact_tail)
        if compact_initials and not compact_tail:
            tokens.extend([x[0] for x in compact_initials])
            continue

        raw = _clean_surface_token(raw0)
        if not raw:
            continue

        # A single initial with a dot: Т. -> Т; after cleaning it is a single uppercase letter.
        if re.fullmatch(r"[A-ZА-ЯЁӞӜӴӦ]", raw):
            tokens.append(raw[0])
            continue

        # Keep word-like tokens, including hyphenated words and abbreviations.
        if re.search(r"[0-9A-Za-zА-Яа-яЁёӝӟӥӧӵӞӜӴӦ]", raw):
            tokens.append(raw)

    return tokens


def _looks_like_initial_or_abbrev(surface: str) -> bool:
    s = _clean_surface_token(surface)
    if not s:
        return False
    if re.fullmatch(r"[A-ZА-ЯЁӞӜӴӦ]", s):
        return True
    if len(s) >= 2 and re.fullmatch(r"[A-ZА-ЯЁӞӜӴӦ0-9]+", s):
        return True
    if re.fullmatch(r"[A-ZА-ЯЁӞӜӴӦ](?:\.[A-ZА-ЯЁӞӜӴӦ])+\.?", s):
        return True
    return False


# ─── 2. UniParser tag mappings ────────────────────────────────────────────────

_TENSE = {"PST": "Past", "PRS": "Present", "FUT": "Future"}
_NUMBER = {"SG": "Singular", "PL": "Plural"}
_CASE = {
    "ACC": "Accusative", "GEN": "Genitive", "DAT": "Dative",
    "LOC": "Locative", "ABL": "Ablative", "ILL": "Illative",
    "EL": "Elative", "INS": "Instrumental", "ADV": "Adverbial",
    "PROL": "Prolative", "DELIM": "Delimitative", "TERM": "Terminative",
    "EGR": "Egressive", "CAR": "Caritive",
}
_VERBFORM = {
    "INF": "Infinitive", "CVB": "Converb", "VN": "VerbalNoun",
    "PTCP": "Participle",
}
_VOICE = {"PASS": "Passive", "CAUS": "Causative", "ACT": "Active"}
_MOOD = {"IMP": "Imperative", "DEB": "Debitive", "HORT": "Hortative"}
_ASPECT = {"ITER": "Iterative", "RES": "Resultative"}
_MISC = {"EVID": "Evidential", "ORD": "Ordinal", "COMP": "Comparative", "ATTR": "Attributive"}

_FEAT_TO_PRED = {
    "Tense": "hasCandidateTense",
    "Number": "hasCandidateNumber",
    "Person": "hasCandidatePerson",
    "Case": "hasCandidateCase",
    "CaseAmbiguous": "hasCandidateCaseAmbiguous",
    "Voice": "hasCandidateVoice",
    "Mood": "hasCandidateMood",
    "VerbForm": "hasCandidateVerbForm",
    "Aspect": "hasCandidateAspect",
    "Polarity": "hasCandidatePolarity",
    "PossPerson": "hasCandidatePossessivePerson",
    "PossNumber": "hasCandidatePossessiveNumber",
    "DerivationalFeature": "hasCandidateDerivationalFeature",
}

# Context lexicons used only to emit explicit ABox cues; rules should still be conservative.
_PERSON_PRONOUNS = {"мон": "1", "тон": "2", "ми": "1", "тӥ": "2"}
_PERSON_NUMBERS = {"мон": "Singular", "тон": "Singular", "ми": "Plural", "тӥ": "Plural"}
_NEG_AUXILIARIES = {"уд", "ум", "уг", "ӧд", "ӧм", "ӧз", "уз", "ӧй"}
_PRESENT_ADVERBS = {"туннэ", "али", "татын", "отын", "ни"}
_QUESTION_WORDS = {"кин", "ма", "кытын", "кыӵе", "куке", "кытчы", "кызьы", "ку", "кӧня"}


def _raw_tag_parts(tag: str) -> list[str]:
    parts = [p for p in (tag or "").split("-") if p]
    if parts and parts[0] == "STEM":
        parts = parts[1:]
    return parts


def _parse_person_number(value: str, features: dict[str, Any]) -> bool:
    """Parse 1SG/2PL/12/PRS.12-like fragments."""
    m = re.fullmatch(r"(12|[123])(SG|PL)?", value)
    if m:
        features["Person"] = m.group(1)
        if m.group(2):
            features["Number"] = _NUMBER[m.group(2)]
        return True
    return False


def _parse_tag(tag: str) -> dict[str, Any]:
    """Parse a raw UniParser tag into normalized features, preserving unknown parts."""
    features: dict[str, Any] = {"RawTagParts": _raw_tag_parts(tag)}

    for part in _raw_tag_parts(tag):
        if not part:
            continue

        # PRS.12, PRS.3PL, FUT.3SG-like fused tense/person forms.
        m_compound = re.fullmatch(r"(PST|PRS|FUT)\.([123]{1,2})(SG|PL)?", part)
        if m_compound:
            features["Tense"] = _TENSE[m_compound.group(1)]
            features["Person"] = m_compound.group(2)
            if m_compound.group(3):
                features["Number"] = _NUMBER[m_compound.group(3)]
            continue

        # NEG.PL / NEG.SG / PTCP.NEG / NEG.ATTR etc.
        if "." in part:
            subparts = [p for p in part.split(".") if p]
            handled_any = False
            for sp in subparts:
                handled_any = _parse_single_tag_part(sp, features) or handled_any
            if handled_any:
                continue

        if _parse_single_tag_part(part, features):
            continue

        # Slash ambiguity: ACC/ILL, INS/LOC etc.
        if "/" in part:
            vals = [x for x in part.split("/") if x]
            mapped = [_CASE.get(x, x) for x in vals]
            features["CaseAmbiguous"] = "/".join(mapped)
            continue

        features.setdefault("UnparsedRawParts", []).append(part)

    return features


def _parse_single_tag_part(part: str, features: dict[str, Any]) -> bool:
    if not part:
        return False

    # Possessives: P.3SG, P.3SG.ACC, P.2PL.ACC, etc. If already split by dots,
    # this may see P and 3SG separately; handle both patterns.
    m_poss = re.fullmatch(r"P\.?(12|[123])?(SG|PL)?", part)
    if m_poss and (m_poss.group(1) or m_poss.group(2)):
        if m_poss.group(1):
            features["PossPerson"] = m_poss.group(1)
        if m_poss.group(2):
            features["PossNumber"] = _NUMBER[m_poss.group(2)]
        return True
    if part == "P":
        features["DerivationalFeature"] = "Possessive"
        return True

    if _parse_person_number(part, features):
        return True

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
    elif part == "NEG":
        features["Polarity"] = "Negative"
    elif part in _MISC:
        # EVID and ATTR are treated as form-like features for disambiguation.
        if part == "EVID":
            features["VerbForm"] = "Evidential"
        elif part == "ATTR":
            features["VerbForm"] = "Attributive"
        else:
            features["DerivationalFeature"] = _MISC[part]
    else:
        return False
    return True


def _infer_pos(features: dict[str, Any], raw_tag: str) -> str:
    parts = set(features.get("RawTagParts", []))
    vf = features.get("VerbForm", "")
    if vf == "VerbalNoun":
        return "Noun"
    if vf in {"Infinitive", "Converb", "Participle", "Evidential"}:
        return "Verb"
    if "Tense" in features or "Person" in features or "Mood" in features or "Voice" in features:
        return "Verb"
    if "Case" in features or "CaseAmbiguous" in features or "PossPerson" in features or "P" in parts:
        return "Noun"
    if raw_tag.strip() == "STEM":
        return "Noun"
    return "Unknown"


def _add_candidate_assertions(assertions: list[dict[str, Any]], ana_id: str, features: dict[str, Any]) -> None:
    # Preserve every raw tag part so LLM-generated rules can use source-grounded raw evidence.
    for part in features.get("RawTagParts", []):
        assertions.append({"predicate": "hasRawTagPart", "args": [ana_id, part]})
        assertions.append({"predicate": "hasCandidateFeatureString", "args": [ana_id, part]})

    for part in features.get("UnparsedRawParts", []):
        assertions.append({"predicate": "hasUnparsedRawTagPart", "args": [ana_id, part]})

    for feat_key, predicate in _FEAT_TO_PRED.items():
        if feat_key in features:
            assertions.append({"predicate": predicate, "args": [ana_id, features[feat_key]]})


def _tokenize_braced_surface(sentence: str) -> list[str]:
    """Return corpus tokens marked as {{...}}; punctuation outside braces is not a token."""
    toks = re.findall(r"\{\{(.*?)\}\}", sentence or "")
    return [_clean_surface_token(t) for t in toks if _clean_surface_token(t)]


def _tokenize_surface_aligned(sentence: str) -> list[str]:
    """Whitespace tokenizer for unbraced data; preserves initials as one token."""
    s = re.sub(r"[{}]", "", sentence or "")
    s = s.replace("\\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    out: list[str] = []
    for raw in re.split(r"\s+", s):
        tok = _clean_surface_token(raw)
        if tok and re.search(r"[0-9A-Za-zА-Яа-яЁёӝӟӥӧӵӞӜӴӦ]", tok):
            out.append(tok)
    return out


_PARTICLE_STEM_FALLBACK = {
    "гинэ", "но", "нош", "ик", "али", "ини", "берло", "одно", "мед", "ук", "на", "со", "таче"
}


def _has_tag(tag_alts: list[str], tag: str) -> bool:
    return any((t or "").strip() == tag for t in tag_alts)


def _has_bare_stem(tag_alts: list[str]) -> bool:
    return _has_tag(tag_alts, "STEM")


def _add_synthetic_tag(tag_alts: list[str], lemma_alts: list[str], tag: str, lemma: str) -> None:
    if not _has_tag(tag_alts, tag):
        tag_alts.append(tag)
        lemma_alts.append(lemma or "*")

def _dedupe_alts_by_raw_tag(
    tag_alts: list[str],
    lemma_alts: list[str],
) -> tuple[list[str], list[str]]:
    """Remove duplicate alternatives with the same raw UniParser tag.

    This is safe for the current evaluation because stage_udmurt_two compares
    selected candidates by hasRawTag, not by candidate ID.

    Example:
        ["STEM", "STEM-PST", "STEM"] -> ["STEM", "STEM-PST"]
    """
    seen: set[str] = set()
    out_tags: list[str] = []
    out_lemmas: list[str] = []

    for i, tag in enumerate(tag_alts or []):
        raw = str(tag or "").strip()
        if not raw or raw in seen:
            continue

        seen.add(raw)
        out_tags.append(raw)

        if i < len(lemma_alts):
            out_lemmas.append(lemma_alts[i])
        else:
            out_lemmas.append("")

    return out_tags, out_lemmas


def _is_wordlike_surface(surface: str) -> bool:
    s = _clean_surface_token(surface)
    return bool(s and re.search(r"[A-Za-zА-Яа-яЁёӝӟӥӧӵӞӜӴӦ]", s))


def _looks_like_nominal_context(surface_tokens: list[str], idx: int) -> bool:
    """High-precision universal cue for borrowed nouns/proper names, not a word-specific hack."""
    cur = _clean_surface_token(surface_tokens[idx] if idx < len(surface_tokens) else "")
    prev = _clean_surface_token(surface_tokens[idx - 1] if idx > 0 else "")
    nxt = _clean_surface_token(surface_tokens[idx + 1] if idx + 1 < len(surface_tokens) else "")
    if not cur:
        return False

    cur_is_lower_word = bool(re.fullmatch(r"[а-яёӝӟӥӧӵ]+", cur))
    cur_is_capitalized = bool(re.fullmatch(r"[A-ZА-ЯЁӞӜӴӦ][A-Za-zА-Яа-яЁёӝӟӥӧӵӞӜӴӦ-]*", cur))
    prev_is_name_like = bool(prev and (prev[:1].isupper() or _looks_like_initial_or_abbrev(prev)))
    next_is_name_like = bool(nxt and (nxt[:1].isupper() or _looks_like_initial_or_abbrev(nxt)))

    # "Кынкорт команда", "Единая Россия партия", surnames/initials lists, organization names.
    if cur_is_lower_word and prev_is_name_like:
        return True
    if cur_is_capitalized and (prev_is_name_like or next_is_name_like):
        return True
    if _looks_like_initial_or_abbrev(cur):
        return True
    return False




def _lemma_raw(lemma_group: list[str]) -> str:
    return " || ".join(x for x in (lemma_group or []) if x).strip()


def _lemma_has_o_suffix(lemma_group: list[str], surface: str = "") -> bool:
    raw = _lemma_raw(lemma_group).lower()
    surf = _clean_surface_token(surface).lower()
    # UniParser breakdown usually marks attributive/future/present -о as a separate segment: ур-о.
    return bool(
        re.search(r"-(?:о|ё)(?:\b|$)", raw)
        or surf.endswith(("о", "ё", "оз", "озы"))
    )


def _tag_group_needs_o_like_lemma(tag_group: list[str]) -> bool:
    tags = set(t.strip() for t in (tag_group or []) if t and t.strip())
    if not tags:
        return False
    return any(
        t == "STEM-ATTR" or t.startswith("STEM-FUT") or t.startswith("STEM-PRS.3PL")
        for t in tags
    )


def _tag_group_has_stem(tag_group: list[str]) -> bool:
    return any((t or "").strip() == "STEM" for t in (tag_group or []))


def _lemma_surface_is_probably_uninflected_name_or_abbrev(lemma_group: list[str], surface: str) -> bool:
    raw = _lemma_raw(lemma_group)
    surf = _clean_surface_token(surface)
    base = surf or raw
    if not base:
        return False
    # No explicit morpheme boundary in the UniParser lemma/breakdown: likely a bare name/acronym/borrowing.
    raw_has_morph_boundary = "-" in raw
    name_like = bool(base[:1].isupper() or _looks_like_initial_or_abbrev(base))
    return name_like and not raw_has_morph_boundary


def _tag_group_compatible_with_lemma(tag_group: list[str], lemma_group: list[str], surface: str) -> bool:
    """Conservative alignment check. False means: probably a missing tag slot before this tag group."""
    if not tag_group:
        return True
    if _tag_group_has_stem(tag_group) and len(set(tag_group)) == 1:
        return True
    if _tag_group_needs_o_like_lemma(tag_group):
        return _lemma_has_o_suffix(lemma_group, surface)
    # For most other tags, keep original order unless we have a very strong reason to insert a gap.
    return True


def _align_lemma_and_tag_groups(
    lemma_groups: list[list[str]],
    tag_groups: list[list[str]],
    surface_tokens: list[str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Align UniParser tag groups to lemma/surface token slots.

    UniParser/social-media rows sometimes contain fewer tag groups than lemma tokens because
    proper names/acronyms are omitted from the tag line. A simple zip then shifts a real
    morphological group onto the wrong token, e.g. STEM-ATTR from ур-о onto УИИЯЛ.

    This alignment does NOT delete candidates. It inserts empty tag slots for obvious
    uninflected names/acronyms when the next tag group is incompatible with that lemma.
    The normal fallback layer then fills those empty slots with synthetic STEM.
    """
    if not lemma_groups:
        return list(tag_groups), [[] for _ in tag_groups]

    aligned_tags: list[list[str]] = []
    aligned_lemmas: list[list[str]] = []
    j = 0
    for i, lemma_group in enumerate(lemma_groups):
        surface = surface_tokens[i] if i < len(surface_tokens) else (_lemma_raw(lemma_group).split("-")[0] if lemma_group else "")
        current_tag = tag_groups[j] if j < len(tag_groups) else []

        if current_tag and not _tag_group_compatible_with_lemma(current_tag, lemma_group, surface):
            if _lemma_surface_is_probably_uninflected_name_or_abbrev(lemma_group, surface):
                # Insert a missing bare-name slot and keep current_tag for the next lemma.
                aligned_tags.append([])
                aligned_lemmas.append(lemma_group)
                continue

        aligned_tags.append(current_tag if j < len(tag_groups) else [])
        aligned_lemmas.append(lemma_group)
        if j < len(tag_groups):
            j += 1

    # If tags remain after all lemmas, append them as analyzer-only slots.
    while j < len(tag_groups):
        aligned_tags.append(tag_groups[j])
        aligned_lemmas.append([])
        j += 1

    return aligned_tags, aligned_lemmas

def _augment_candidate_tags(
    tag_alts: list[str],
    lemma_alts: list[str],
    surface_tokens: list[str],
    tok_idx: int,
) -> tuple[list[str], list[str]]:
    """Add fallback candidates without deleting UniParser candidates.

    This is candidate generation repair, not disambiguation. It makes POS-like
    options available to disambiguate(), but does not force the final choice.
    """
    tag_alts = list(tag_alts or [])
    lemma_alts = list(lemma_alts or [])
    surface = _clean_surface_token(surface_tokens[tok_idx] if tok_idx < len(surface_tokens) else "")
    lower = surface.lower()

    if not _is_wordlike_surface(surface):
        return tag_alts, lemma_alts

    # Universal unknown-token fallback: a word token with no UniParser tags should still have a candidate.
    if not tag_alts:
        _add_synthetic_tag(tag_alts, lemma_alts, "STEM", surface)

    # Lexical particles/adverbs are often uninflected STEM.
    if lower in _PARTICLE_STEM_FALLBACK and not _has_bare_stem(tag_alts):
        _add_synthetic_tag(tag_alts, lemma_alts, "STEM", surface)

    # Proper names, borrowed nouns, appositions: add a noun-like bare STEM option,
    # but only when the existing analyses are absent or implausible for names.
    # Do not add STEM to well-formed single analyses such as STEM-EL, STEM-VN, STEM-COMP, etc.
    if _looks_like_nominal_context(surface_tokens, tok_idx) and not _has_bare_stem(tag_alts):
        unique_tags = set(t.strip() for t in tag_alts if t and t.strip())
        existing_tags_look_bad_for_name = (
            not unique_tags
            or all(
                ("IMP" in t or "PRS" in t or "FUT" in t or t == "STEM-ATTR" or t.startswith("STEM-P.1"))
                for t in unique_tags
            )
        )
        if existing_tags_look_bad_for_name:
            _add_synthetic_tag(tag_alts, lemma_alts, "STEM", surface)

    # Productive suffix backups for common social-media omissions.
    if lower.endswith("сько") and not any("PRS.12" in t for t in tag_alts):
        _add_synthetic_tag(tag_alts, lemma_alts, "STEM-PRS.12", surface)

    if (lower.endswith("оз") or lower.endswith("оз-а")) and not any("FUT-3SG" in t for t in tag_alts):
        _add_synthetic_tag(tag_alts, lemma_alts, "STEM-FUT-3SG", surface)

    return tag_alts, lemma_alts


def uniparser_to_abox(sentence: str, uniparser_output: str, sent_id: str = "sent1") -> dict:
    """Convert one UniParser output line into the ABox expected by disambiguate()."""
    uniparser_output = (uniparser_output or "").strip()
    if "\t" in uniparser_output:
        lemma_line, tag_line = uniparser_output.split("\t", 1)
    else:
        lemma_line, tag_line = "", uniparser_output

    raw_tag_groups = _split_to_groups(tag_line)
    raw_lemma_groups = _split_to_groups(lemma_line) if lemma_line else []

    # Prefer corpus {{...}} tokens or UniParser lemma groups for alignment.
    # Do not let punctuation-only surface tokens such as dashes shift candidates.
    surface_tokens = _tokenize_braced_surface(sentence) or _tokenize_surface_aligned(sentence)

    tag_groups, lemma_groups = _align_lemma_and_tag_groups(raw_lemma_groups, raw_tag_groups, surface_tokens)

    # Aligned analyzer slots define ABox tokens. Surface is context, not authority for creating extra slots.
    n = max(len(tag_groups), len(lemma_groups))
    if n == 0:
        n = len(surface_tokens)

    individuals: dict[str, dict[str, Any]] = {sent_id: {"type": "Sentence"}}
    assertions: list[dict[str, Any]] = [{"predicate": "hasText", "args": [sent_id, sentence]}]

    for tok_idx in range(n):
        tok_id = f"tok{tok_idx + 1}"
        surface = surface_tokens[tok_idx] if tok_idx < len(surface_tokens) else ""
        if not surface and tok_idx < len(lemma_groups) and lemma_groups[tok_idx]:
            surface = lemma_groups[tok_idx][0].split("-")[0]
        surface = _clean_surface_token(surface)

        individuals[tok_id] = {"type": "Token"}
        assertions.extend([
            {"predicate": "belongsToSentence", "args": [tok_id, sent_id]},
            {"predicate": "hasForm", "args": [tok_id, surface]},
            {"predicate": "hasPosition", "args": [tok_id, str(tok_idx + 1)]},
        ])

        if _looks_like_initial_or_abbrev(surface):
            assertions.append({"predicate": "tokenLooksLikeAbbreviation", "args": [tok_id]})

        tag_alts = tag_groups[tok_idx] if tok_idx < len(tag_groups) else []
        lemma_alts = lemma_groups[tok_idx] if tok_idx < len(lemma_groups) else []

        tag_alts, lemma_alts = _augment_candidate_tags(tag_alts, lemma_alts, surface_tokens, tok_idx)
        tag_alts, lemma_alts = _dedupe_alts_by_raw_tag(tag_alts, lemma_alts)

        for alt_idx, tag_raw in enumerate(tag_alts):
            ana_id = f"ana_tok{tok_idx + 1}_alt{alt_idx + 1}"
            lemma_raw = lemma_alts[alt_idx].strip() if alt_idx < len(lemma_alts) else surface
            lemma_clean = lemma_raw.split("-")[0] if "-" in lemma_raw else lemma_raw
            tag_raw = tag_raw.strip()
            if not tag_raw:
                continue

            features = _parse_tag(tag_raw)
            pos = _infer_pos(features, tag_raw)

            individuals[ana_id] = {"type": "CandidateAnalysis"}
            assertions.extend([
                {"predicate": "candidateOf", "args": [ana_id, tok_id]},
                {"predicate": "hasLemma", "args": [ana_id, lemma_clean]},
                {"predicate": "hasMorphBreakdown", "args": [ana_id, lemma_raw]},
                {"predicate": "hasRawTag", "args": [ana_id, tag_raw]},
            ])
            if pos != "Unknown":
                assertions.append({"predicate": "hasCandidatePOS", "args": [ana_id, pos]})
            _add_candidate_assertions(assertions, ana_id, features)

    tok_ids = [f"tok{i + 1}" for i in range(n)]
    for i in range(len(tok_ids) - 1):
        assertions.append({"predicate": "precedesToken", "args": [tok_ids[i], tok_ids[i + 1]]})
    if tok_ids:
        assertions.append({"predicate": "tokenIsClauseFinal", "args": [tok_ids[-1], True]})

    surfaces_lower = []
    for i in range(n):
        surf = surface_tokens[i] if i < len(surface_tokens) else ""
        surfaces_lower.append(_clean_surface_token(surf).lower())

    has_neg_aux = False
    for i, surf in enumerate(surfaces_lower):
        tok_id = f"tok{i + 1}"
        if surf in _PERSON_PRONOUNS:
            assertions.append({"predicate": "tokenHasPerson", "args": [tok_id, _PERSON_PRONOUNS[surf]]})
            assertions.append({"predicate": "tokenHasNumber", "args": [tok_id, _PERSON_NUMBERS.get(surf, "Singular")]})
        if surf in _NEG_AUXILIARIES:
            assertions.append({"predicate": "clauseContainsNegativeAuxiliary", "args": [sent_id, tok_id]})
            has_neg_aux = True
        if surf in _PRESENT_ADVERBS:
            assertions.append({"predicate": "hasPresentTimeAdverb", "args": [sent_id, tok_id]})
        if surf in _QUESTION_WORDS:
            assertions.append({"predicate": "isInterrogativeSentence", "args": [sent_id]})

    if not has_neg_aux:
        assertions.append({"predicate": "clauseHasNoAuxiliary", "args": [sent_id]})

    return {"individuals": individuals, "assertions": assertions}


# ─── Explanations ─────────────────────────────────────────────────────────────

def explain_disambiguation(disamb_result: dict, abox: dict) -> list[dict]:
    raw_tags: dict[str, str] = {}
    surfaces: dict[str, str] = {}
    for a in abox.get("assertions", []):
        pred = a.get("predicate")
        args = a.get("args", [])
        if pred == "hasRawTag" and len(args) == 2:
            raw_tags[args[0]] = args[1]
        elif pred in {"hasForm", "hasSurfaceForm"} and len(args) == 2:
            surfaces[args[0]] = args[1]

    results = []
    for tok_id, info in (disamb_result or {}).get("tokens", {}).items():
        surface = surfaces.get(tok_id, tok_id)
        status = info.get("status", "unknown") if isinstance(info, dict) else "unknown"
        remaining = info.get("remaining_candidates", []) if isinstance(info, dict) else []
        rejected = info.get("rejected_candidates", []) if isinstance(info, dict) else []
        all_cands = list(dict.fromkeys(remaining + rejected))
        was_ambiguous = len(all_cands) > 1
        selected = info.get("selected_candidate") if isinstance(info, dict) else None
        selected_tag = raw_tags.get(selected) if selected else None
        rejected_tags = [raw_tags.get(c, c) for c in rejected]
        all_tags = [raw_tags.get(c, c) for c in all_cands]

        reason_parts = []
        explanations = info.get("explanations", {}) if isinstance(info, dict) else {}
        for cand_id, expl in explanations.items():
            tag = raw_tags.get(cand_id, cand_id)
            supporting = expl.get("supporting_rules", []) if isinstance(expl, dict) else []
            rejecting = expl.get("rejecting_constraints", []) if isinstance(expl, dict) else []
            if rejecting:
                reason_parts.append(f"[{tag}] отклонён: {'; '.join(map(str, rejecting))}")
            elif supporting and cand_id == selected:
                reason_parts.append(f"[{tag}] выбран: {'; '.join(map(str, supporting))}")

        if status == "unresolved":
            reason_parts.append("Правил недостаточно для однозначного выбора")
        elif status == "contradiction":
            reason_parts.append("Все кандидаты отклонены — возможно противоречивые ограничения")

        results.append({
            "token_id": tok_id,
            "surface": surface,
            "status": status,
            "was_ambiguous": was_ambiguous,
            "selected_tag": selected_tag,
            "rejected_tags": rejected_tags,
            "all_candidates": all_tags,
            "reason": " | ".join(reason_parts) if reason_parts else ("однозначный анализ" if not was_ambiguous else ""),
        })
    return results


# ─── Utilities for corpus conversion ──────────────────────────────────────────

def parse_corpus_file(filepath: str) -> list[dict]:
    results = []
    path = Path(filepath)
    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 3:
                continue
            source = parts[0].strip('"').strip()
            sentence_col = None
            analysis_col = None
            for i, col in enumerate(parts):
                if "{{" in col:
                    sentence_col = i
                    for j in range(i + 1, len(parts)):
                        if parts[j].strip():
                            analysis_col = j
                            break
                    break
            if sentence_col is None:
                continue
            raw_sentence = parts[sentence_col].strip()
            sentence = re.sub(r"[{}]", "", raw_sentence).strip()
            uniparser_output = parts[analysis_col].strip().strip('"') if analysis_col is not None else ""
            if not sentence or not uniparser_output:
                continue
            results.append({"sent_id": f"sent_{line_no}", "sentence": sentence, "uniparser_output": uniparser_output, "source": source})
    return results


def corpus_to_aboxes(filepath: str) -> list[dict]:
    records = parse_corpus_file(filepath)
    return [{"sent_id": r["sent_id"], "sentence": r["sentence"], "source": r["source"], "abox": uniparser_to_abox(r["sentence"], r["uniparser_output"], r["sent_id"])} for r in records]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
        aboxes = corpus_to_aboxes(corpus_path)
        out = corpus_path.replace(".csv", "_aboxes.json")
        Path(out).write_text(json.dumps(aboxes, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Готово: {len(aboxes)} предложений → {out}")
    else:
        example = ("Озьы ик удмуртъёс пӧлын Петров Э.С., РАН.", "озьы ик удмурт-ъёс пӧл-ын петров Э С РАН\tSTEM STEM STEM-PL STEM-LOC STEM STEM STEM STEM")
        print(json.dumps(uniparser_to_abox(example[0], example[1]), ensure_ascii=False, indent=2))
