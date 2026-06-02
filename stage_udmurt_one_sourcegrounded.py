#!/usr/bin/env python3
"""
stage_udmurt_one_sourcegrounded.py
==================================

Source-grounded Stage 1 for the Udmurt SOLAR thesis project.

This version gives the LLM three evidence layers:
1) a strict ABox/interpreter contract;
2) source rules from UniParser CG3, if --cg3 is provided;
3) train-derived ambiguity groups from train.csv, if --train is provided.

It is designed to reduce hallucinated grammar rules and to force the generated
interpreter to return token-level decisions that stage_udmurt_two.py can convert
into gold_tag_line-like strings.
"""
from __future__ import annotations

import json
import os
import re
from argparse import ArgumentParser
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from solar.knowledge_acquisition import predict

PIPE = "\x00"


# ─── Model setup ──────────────────────────────────────────────────────────────

@cache
def chat_model(model: str, no_cache: bool) -> BaseChatModel:
    if not no_cache:
        set_llm_cache(SQLiteCache(database_path="./.cache_acquisition"))

    aliases = {
        "claude": "anthropic/claude-opus-4.7",
        "opus": "anthropic/claude-opus-4.7",
        "opus-4.7": "anthropic/claude-opus-4.7",
        "opus-4.6": "anthropic/claude-opus-4.6",
        "gpt-5.1": "openai/gpt-5.1",
        "gpt-5-mini": "openai/gpt-5-mini",
        "qwen235b": "qwen/qwen3-235b-a22b-2507",
        "qwen32b": "qwen/qwen3-32b",
        "gpt-5.4": "openai/gpt-5.4",
        "sonnet": "anthropic/claude-sonnet-4.6",
    }
    resolved_model = aliases.get(model, model)

    if "/" in resolved_model:
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=resolved_model,
            temperature=0.0,
        )

    if model == "local_qwen":
        return ChatOpenAI(
            base_url=os.getenv("LOCAL_QWEN_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("LOCAL_QWEN_API_KEY") or os.getenv("NID_TOKEN") or "EMPTY",
            model=os.getenv("LOCAL_QWEN_MODEL", "Qwen3:32b"),
            temperature=0.0,
        )

    raise ValueError(
        f"Unknown model '{model}'. Use claude, gpt-5.1, gpt-5-mini, qwen235b, "
        f"qwen32b, local_qwen, or a full OpenRouter id."
    )


def safe_model_name(model: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model.strip())
    return safe.strip("_") or "model"


# ─── Train profile extraction ────────────────────────────────────────────────

def extract_tag_line(cands_analtag_line: str) -> str:
    text = (cands_analtag_line or "").strip()
    if "\t" in text:
        return text.rsplit("\t", 1)[1].strip()
    if "\\t" in text:
        return text.rsplit("\\t", 1)[1].strip()
    return text


def split_candidate_tag_groups(tag_line: str) -> list[list[str]]:
    text = (tag_line or "").strip()
    if not text:
        return []
    text = re.sub(r"\s*\|\|\s*", PIPE, text)
    return [g.split(PIPE) for g in text.split() if g]


def normalize_sentence(tokens_braced: str) -> str:
    s = re.sub(r"[{}]", "", str(tokens_braced or ""))
    s = s.replace("\\n", " ")
    return re.sub(r"\s+", " ", s).strip()


def make_train_profile(train_csv: str, max_groups: int, max_examples_per_group: int) -> dict[str, Any]:
    df = pd.read_csv(train_csv, sep=";", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    required = {"tokens_braced", "cands_analtag_line", "gold_tag_line"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing train columns: {missing}")

    candidate_tags = Counter()
    gold_tags = Counter()
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    length_mismatches = []

    for row_idx, row in df.iterrows():
        sentence = normalize_sentence(row.get("tokens_braced", ""))
        gold_line = str(row.get("gold_tag_line", "")).strip()
        gold_tokens = gold_line.split()
        tag_line = extract_tag_line(str(row.get("cands_analtag_line", "")))
        cand_groups = split_candidate_tag_groups(tag_line)

        if len(gold_tokens) != len(cand_groups):
            length_mismatches.append({
                "row_index": int(row_idx),
                "sentence": sentence,
                "gold_token_count": len(gold_tokens),
                "candidate_token_count": len(cand_groups),
                "gold_tag_line": gold_line,
                "uniparser_tag_line": tag_line,
            })

        n = min(len(gold_tokens), len(cand_groups))
        for i in range(n):
            unique = tuple(dict.fromkeys(cand_groups[i]))
            gold = gold_tokens[i]
            for c in unique:
                candidate_tags[c] += 1
            gold_tags[gold] += 1
            if len(unique) <= 1:
                continue
            rec = groups.setdefault(unique, {
                "candidates": list(unique),
                "count": 0,
                "gold_counts": Counter(),
                "examples": [],
            })
            rec["count"] += 1
            rec["gold_counts"][gold] += 1
            if len(rec["examples"]) < max_examples_per_group:
                rec["examples"].append({
                    "row_index": int(row_idx),
                    "token_index": i,
                    "sentence": sentence,
                    "candidates": list(unique),
                    "gold": gold,
                    "gold_tag_line": gold_line,
                    "uniparser_tag_line": tag_line,
                })

    sorted_groups = sorted(groups.values(), key=lambda x: (-x["count"], x["candidates"]))[:max_groups]
    for g in sorted_groups:
        g["gold_counts"] = dict(g["gold_counts"].most_common())

    return {
        "rows": int(len(df)),
        "unique_candidate_tags": len(candidate_tags),
        "unique_gold_tags": len(gold_tags),
        "candidate_tags_top": dict(candidate_tags.most_common(160)),
        "gold_tags_top": dict(gold_tags.most_common(160)),
        "ambiguity_group_count": len(groups),
        "ambiguity_groups_top": sorted_groups,
        "length_mismatch_count": len(length_mismatches),
        "length_mismatches_sample": length_mismatches[:12],
    }


def train_profile_to_prompt(profile: dict[str, Any]) -> str:
    lines = [
        "# TRAIN-DERIVED AMBIGUITY PROFILE",
        "",
        "This section is derived automatically from train.csv.",
        "Use it to infer general disambiguation behavior from cands_analtag_line -> gold_tag_line.",
        "Do NOT hard-code sentence IDs or memorise exact sentences.",
        "At test time gold_tag_line is unavailable.",
        "",
        f"Training rows: {profile['rows']}",
        f"Unique candidate tags: {profile['unique_candidate_tags']}",
        f"Unique gold tags: {profile['unique_gold_tags']}",
        f"Observed ambiguity groups: {profile['ambiguity_group_count']}",
        f"Length mismatch rows: {profile['length_mismatch_count']}",
        "",
        "## Important note about initials and abbreviations",
        "Initials and abbreviations are common in the social-media corpus. If a token is an initial/acronym and has exactly one STEM candidate, it should be resolved as STEM. Do not drop such tokens.",
        "",
        "## Most frequent observed ambiguity groups",
    ]
    for idx, g in enumerate(profile["ambiguity_groups_top"], 1):
        lines.append("")
        lines.append(f"### Ambiguity group {idx}")
        lines.append("Candidates: " + " || ".join(g["candidates"]))
        lines.append("Observed gold counts: " + json.dumps(g["gold_counts"], ensure_ascii=False))
        for ex in g["examples"]:
            lines.append("Example:")
            lines.append(f"  sentence: {ex['sentence']}")
            lines.append(f"  token_index: {ex['token_index']}")
            lines.append(f"  candidates: {' || '.join(ex['candidates'])}")
            lines.append(f"  gold: {ex['gold']}")
            lines.append(f"  gold_tag_line: {ex['gold_tag_line']}")
    return "\n".join(lines) + "\n"


# ─── Prompt contract ─────────────────────────────────────────────────────────

SOURCE_GROUNDED_CONTRACT = r'''
# CRITICAL SOURCE-GROUNDED IMPLEMENTATION CONTRACT

You are generating a Python interpreter for Udmurt morphological annotation.
The goal is to select candidate analyses produced by UniParser and to return a token-level decision structure.

Do NOT invent linguistic rules. Use only:
1. the exact ABox schema below;
2. source rules from UniParser CG3 if provided;
3. train-derived ambiguity patterns if provided;
4. the explicitly given grammar notes.
If a case is not supported by these sources, keep it unresolved rather than guessing.

The generated Python file MUST define exactly:

def disambiguate(abox: dict, sentence_id: str) -> dict:

The function MUST NOT return a number as the main result.
The function MUST NOT return only a tag string.
The function MUST return:

{
  "sentence_id": sentence_id,
  "tokens": {
    token_id: {
      "status": "resolved" | "unresolved" | "contradiction" | "no_candidates",
      "selected_candidate": candidate_id or None,
      "remaining_candidates": list,
      "rejected_candidates": list,
      "explanations": dict
    }
  }
}

Every Token linked by belongsToSentence(token_id, sentence_id) MUST appear in result["tokens"].
Never return an empty tokens dictionary if the ABox contains target-sentence tokens.

Decision policy:
- zero candidates -> no_candidates;
- one candidate -> resolved, selected_candidate = that candidate;
- multiple candidates -> apply source-grounded rules;
- reject only on explicit contradictory evidence;
- if exactly one candidate remains -> resolved;
- if multiple candidates remain and exactly one has source-grounded support -> resolved;
- otherwise unresolved.

The downstream evaluator converts selected_candidate to hasRawTag and compares a whitespace-separated tag sequence with gold_tag_line, e.g.:
STEM STEM-FUT-1PL STEM

# EXACT ABox predicates emitted by the parser

Structural:
- belongsToSentence(token_id, sentence_id)
- candidateOf(candidate_id, token_id)
- hasForm(token_id, surface_string)
- hasPosition(token_id, position_string)
- precedesToken(token_a, token_b)

Candidate raw evidence:
- hasLemma(candidate_id, lemma_string)
- hasMorphBreakdown(candidate_id, morph_breakdown_string)
- hasRawTag(candidate_id, raw_tag_string)
- hasRawTagPart(candidate_id, tag_part_string)
- hasCandidateFeatureString(candidate_id, tag_part_string)
- hasUnparsedRawTagPart(candidate_id, tag_part_string)

Candidate normalized features:
- hasCandidatePOS(candidate_id, POS_string)
- hasCandidateCase(candidate_id, Case_string)
- hasCandidateCaseAmbiguous(candidate_id, Case_string)
- hasCandidateNumber(candidate_id, Number_string)
- hasCandidatePerson(candidate_id, Person_string)
- hasCandidateTense(candidate_id, Tense_string)
- hasCandidateMood(candidate_id, Mood_string)
- hasCandidateVoice(candidate_id, Voice_string)
- hasCandidateAspect(candidate_id, Aspect_string)
- hasCandidateVerbForm(candidate_id, VerbForm_string)
- hasCandidatePolarity(candidate_id, Polarity_string)
- hasCandidatePossessivePerson(candidate_id, person_string)
- hasCandidatePossessiveNumber(candidate_id, number_string)
- hasCandidateDerivationalFeature(candidate_id, feature_string)

Context:
- tokenLooksLikeAbbreviation(token_id)
- hasSubject(predicate_token_id, subject_token_id)
- tokenHasPerson(token_id, "1" | "2" | "3")
- tokenHasNumber(token_id, "Singular" | "Plural")
- tokenHasAnimacy(token_id, "Animate" | "Inanimate")
- tokenHasSemanticRole(token_id, "Agent" | "Patient")
- clauseHasNoAuxiliary(sentence_or_clause_id)
- clauseContainsNegativeAuxiliary(sentence_or_clause_id, aux_token_id)
- tokenIsClauseFinal(token_id, boolean)
- tokenIsConjunct(token_id)
- precedingConjunct(token_id, previous_token_id)
- hasPresentTimeAdverb(sentence_id, token_id)
- isInterrogativeSentence(sentence_id)

Do NOT rely on unsupported renamed predicates such as: precedes, inClause, hasTokenPerson, hasTagString, hasNonFiniteForm.
You may support them only as optional aliases, never as the main schema.

# Feature values emitted by the parser
Tense: Past, Present, Future
Number: Singular, Plural
Person: 1, 2, 3, 12
Voice: Passive, Causative, Active
VerbForm: Infinitive, Converb, VerbalNoun, Participle, Evidential, Attributive
Mood: Imperative, Debitive, Hortative
Aspect: Iterative, Resultative
Polarity: Negative

Preserve raw tag evidence. When in doubt, inspect hasRawTag and hasRawTagPart.
'''


def read_optional_file(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Optional source file not found: {p}")
    return p.read_text(encoding="utf-8")


def build_sourcegrounded_prompt(grammar_text: str, cg3_text: str = "", train_profile_text: str = "") -> str:
    parts = [SOURCE_GROUNDED_CONTRACT]
    if cg3_text:
        parts.extend([
            "\n# SOURCE: UniParser Udmurt Constraint Grammar rules (udmurt_disambiguation.cg3)\n",
            "These are source rules. Convert their SELECT/REMOVE logic conservatively when possible. Do not invent rules beyond them.\n",
            cg3_text[:45000],
        ])
    if train_profile_text:
        parts.extend([
            "\n# SOURCE: train.csv ambiguity profile\n",
            train_profile_text[:55000],
        ])
    parts.extend([
        "\n# SOURCE: Udmurt grammar notes supplied by the researcher\n",
        grammar_text,
    ])
    return "\n\n".join(parts)


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_generated_interpreter(code: str, expected_function: str = "disambiguate") -> list[str]:
    warnings: list[str] = []
    required = [
        f"def {expected_function}", '"tokens"', '"selected_candidate"', '"remaining_candidates"',
        '"resolved"', '"unresolved"', 'belongsToSentence', 'candidateOf', 'hasRawTag',
        'precedesToken',
    ]
    for frag in required:
        if frag not in code:
            warnings.append(f"missing expected fragment: {frag}")
    if "def calculate" in code and f"def {expected_function}" not in code:
        warnings.append("calculate() was generated without disambiguate(); this is incompatible")
    suspicious = ['"precedes"', '"inClause"', '"hasTokenPerson"', '"hasTagString"', '"hasNonFiniteForm"']
    for frag in suspicious:
        if frag in code:
            warnings.append(f"suspicious non-parser predicate name: {frag}")
    return warnings


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = ArgumentParser()
    p.add_argument("--model", default="claude")
    p.add_argument("--statute", default="./eval/udmurt_grammar.txt")
    p.add_argument("--cg3", default=None, help="Path to udmurt_disambiguation.cg3 downloaded from UniParser repo")
    p.add_argument("--train", default=None, help="Path to train.csv for source-grounded ambiguity profile")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--output-prefix", default="")
    p.add_argument("--output-tbox", default=None)
    p.add_argument("--output-py", default=None)
    p.add_argument("--expected-function", default="disambiguate")
    p.add_argument("--max-train-groups", type=int, default=80)
    p.add_argument("--max-examples-per-group", type=int, default=3)
    p.add_argument("--save-prompt", default=None, help="Save the full generated Stage 1 prompt for debugging")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def build_output_paths(args) -> tuple[Path, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_model_name(args.model)
    if args.output_prefix:
        prefix = f"{args.output_prefix}_{prefix}"
    tbox = Path(args.output_tbox) if args.output_tbox else out_dir / f"tbox_{prefix}.json"
    py = Path(args.output_py) if args.output_py else out_dir / f"interpreter_{prefix}.py"
    tbox.parent.mkdir(parents=True, exist_ok=True)
    py.parent.mkdir(parents=True, exist_ok=True)
    return tbox, py


def main():
    load_dotenv()
    args = parse_args()

    statute_path = Path(args.statute)
    if not statute_path.exists():
        raise FileNotFoundError(f"Grammar/statute file not found: {statute_path}")
    grammar_text = statute_path.read_text(encoding="utf-8")
    cg3_text = read_optional_file(args.cg3)

    train_profile_text = ""
    train_profile = None
    if args.train:
        train_profile = make_train_profile(args.train, args.max_train_groups, args.max_examples_per_group)
        train_profile_text = train_profile_to_prompt(train_profile)

    prompt_text = build_sourcegrounded_prompt(grammar_text, cg3_text, train_profile_text)

    if args.save_prompt:
        Path(args.save_prompt).write_text(prompt_text, encoding="utf-8")

    output_tbox, output_py = build_output_paths(args)

    print("── Stage 1: source-grounded knowledge acquisition ─────")
    print(f"Model:       {args.model}")
    print(f"Grammar:     {statute_path}")
    print(f"CG3:         {args.cg3 or '(not provided)'}")
    print(f"Train:       {args.train or '(not provided)'}")
    print(f"TBox output: {output_tbox}")
    print(f"Py output:   {output_py}")
    if train_profile:
        print(f"Train ambiguity groups: {train_profile['ambiguity_group_count']}")

    tbox, tbox_eval_code = predict(chat_model(args.model, args.no_cache), prompt_text, args.debug)

    output_tbox.write_text(json.dumps(tbox.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    output_py.write_text(tbox_eval_code, encoding="utf-8")

    warnings = validate_generated_interpreter(tbox_eval_code, args.expected_function)
    if warnings:
        print("\n[!] Static interpreter-contract warnings:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(f"\nOK: generated interpreter appears compatible with `{args.expected_function}` contract.")

    print("\nNext step:")
    print(f"  uv run stage_udmurt_two_sourcegrounded.py --corpus ./eval/test.csv --tbox-path {output_tbox} --tbox-interpreter-path {output_py} --model {args.model} --output-dir {Path(args.output_dir)} --debug --limit -1")


if __name__ == "__main__":
    main()
