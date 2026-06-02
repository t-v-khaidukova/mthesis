import json
import re
import traceback
from functools import partial
from typing import Final, Type, TypeVar

import pandas as pd
from langchain.chat_models.base import BaseChatModel
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field

from .model import ExtractionEnvelope, TBox, TBoxClass, TBoxProperty, TBoxRule
from .knowledge_application import predict as knowledge_application_predict
from .prompts import (
    CODE_GENERATION_PROMPT,
    CONCEPT_EXTRACTION_PROMPT,
    RULE_FORMULATION_PROMPT,
    RULE_INTEGRATION_PROMPT,
)
from .validation import validate_tbox


MAX_FIX_ATTEMPTS: Final[int] = 3

T = TypeVar("T", bound=BaseModel)


import time

import unicodedata
import re
import json


def sanitize_text(text: str) -> str:
    """
    FIX 1: removes broken surrogate unicode that crashes Gemini / protobuf
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\ud800-\udfff]', '', text)  # 🔥 FIX CRITICAL
    return text


def sanitize_obj(obj):
    """
    FIX 2: recursively clean dict/list structures before json.dumps()
    """
    if isinstance(obj, dict):
        return {k: sanitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_obj(v) for v in obj]
    if isinstance(obj, str):
        return sanitize_text(obj)
    return obj

def invoke_structured(
    llm: BaseChatModel,
    prompt_value: str,
    schema: Type[T],
    max_retries: int = 4,
    base_delay: float = 5.0,
) -> T:
    """
    Вызывает LLM и парсит результат в Pydantic-схему.

    - Попытка 1: нативный structured output.
    - При любой ошибке: откат на llm.invoke() + ручной JSON-парсинг.
    - Транзиентные сетевые ошибки (таймауты, 5xx) повторяются с
      exponential backoff (5s, 10s, 20s, 40s).
    """
    # 🔥 FIX 3: sanitize prompt BEFORE sending to any LLM
    prompt_value = sanitize_text(prompt_value)

    _TRANSIENT_CODES = {524, 502, 503, 504, 429}

    def _is_transient(exc: Exception) -> bool:
        msg = str(exc).lower()
        if any(str(c) in msg for c in _TRANSIENT_CODES):
            return True
        if "timeout" in msg or "timed out" in msg or "connection" in msg:
            return True
        return False

    def _parse_text(text: str) -> T:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())
        return schema.model_validate_json(text)

    last_exc: Exception | None = None

    for attempt in range(max_retries):
        delay = base_delay * (2 ** attempt)   # 5, 10, 20, 40 сек

        # --- попытка A: нативный structured output ---
        try:
            result = llm.with_structured_output(schema).invoke(prompt_value)
            if result is not None:
                return result
        except Exception as e:
            if _is_transient(e):
                last_exc = e
                print(f"[invoke_structured] transient error (structured), "
                      f"retry {attempt + 1}/{max_retries} in {delay:.0f}s: {e}")
                time.sleep(delay)
                continue
            # не транзиентная — пробуем fallback без retry
            pass

        # --- попытка B: plain llm.invoke() + ручной парсинг ---
        try:
            raw = llm.invoke(prompt_value)
            text: str = raw.content if hasattr(raw, "content") else str(raw)
            return _parse_text(text)
        except Exception as e:
            if _is_transient(e):
                last_exc = e
                print(f"[invoke_structured] transient error (plain), "
                      f"retry {attempt + 1}/{max_retries} in {delay:.0f}s: {e}")
                time.sleep(delay)
                continue
            raise   # не транзиентная — пробрасываем сразу

    raise RuntimeError(
        f"[invoke_structured] все {max_retries} попыток исчерпаны. "
        f"Последняя ошибка: {last_exc}"
    )


# ---------------------------------------------------------------------------
# State & helper models
# ---------------------------------------------------------------------------

class TrainingProblem(BaseModel):
    description: str
    query: str
    extracted_abox: dict | None = None
    answer: float | None = None
    error: str | None = None


class State(BaseModel):
    raw_statute_text: str
    candidate_classes: list[ExtractionEnvelope[TBoxClass]] | None = None
    candidate_properties: list[ExtractionEnvelope[TBoxProperty]] | None = None
    candidate_rules: list[ExtractionEnvelope[TBoxRule]] | None = None
    tbox: TBox | None = None
    tbox_validation_issues: list[str] | None = None
    tbox_eval_code: str | None = None
    fix_attempts: int = 0
    human_feedback: str | None = None
    training_evaluation_error: TrainingProblem | None = None


class ConceptExtractionResult(BaseModel):
    classes: list[ExtractionEnvelope[TBoxClass]] = Field(
        ..., description="List of extracted classes."
    )
    properties: list[ExtractionEnvelope[TBoxProperty]] = Field(
        ..., description="List of extracted properties."
    )


class RuleFormulationResult(BaseModel):
    rules: list[ExtractionEnvelope[TBoxRule]] = Field(
        ...,
        description="List of formulated rules, where each 'object' is a TBoxRule.",
    )


class RuleIntegrationResult(BaseModel):
    tbox: TBox = Field(..., description="Final TBox.")


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def concept_extraction_node(state: State, llm: BaseChatModel, debug: bool):
    if debug:
        print("E concept_extraction_node")

    statute = state.raw_statute_text.strip() if state.raw_statute_text else None
    if not statute:
        return {
            "candidate_classes": [],
            "candidate_properties": [],
        }

    # CONCEPT_EXTRACTION_PROMPT использует template_format="jinja2"
    # и переменные: resource_text, source_name, source_type
    prompt_value = CONCEPT_EXTRACTION_PROMPT.format(
        resource_text=statute,
        source_name="udmurt_grammar",
        source_type="grammar",
    )

    res = invoke_structured(llm, prompt_value, ConceptExtractionResult)

    if debug:
        print("X concept_extraction_node")

    return {
        "candidate_classes": res.classes,
        "candidate_properties": res.properties,
    }


def rule_formulation_node(state: State, llm: BaseChatModel, debug: bool):
    if debug:
        print("E rule_formulation_node")

    statute = state.raw_statute_text.strip() if state.raw_statute_text else None
    if not statute:
        return {"candidate_rules": None}

    # RULE_FORMULATION_PROMPT использует str.format() (template_format по умолчанию)
    # и переменные: resource_text, source_name, source_type
    prompt_value = RULE_FORMULATION_PROMPT.format(
        resource_text=statute,
        source_name="udmurt_grammar",
        source_type="grammar",
    )

    res = invoke_structured(llm, prompt_value, RuleFormulationResult)

    if debug:
        print("X rule_formulation_node")

    return {"candidate_rules": res.rules}


def rule_integration_node(state: State, llm: BaseChatModel, debug: bool):
    if debug:
        label = (
            f" (fix attempt #{state.fix_attempts})" if state.fix_attempts > 0 else ""
        )
        print(f"E rule_integration_node{label}")

    prompt_value = RULE_INTEGRATION_PROMPT.format(
        candidate_classes=json.dumps(
            sanitize_obj([cls.model_dump() for cls in (state.candidate_classes or [])])
        ),
        candidate_properties=json.dumps(
            sanitize_obj([prop.model_dump() for prop in (state.candidate_properties or [])])
        ),
        candidate_rules=json.dumps(
            sanitize_obj([rule.model_dump() for rule in (state.candidate_rules or [])])
        ),
        last_tbox=(
            json.dumps(sanitize_obj(state.tbox.model_dump()))
            if state.tbox else None
        ),
        tbox_validation_issues=(
            "\n".join(
                f"  - {issue.strip()}"
                for issue in (state.tbox_validation_issues or [])
            )
            if state.tbox_validation_issues
            else None
        ),
        human_feedback=state.human_feedback,
    )

    # 🔥 FIX 4: final safety layer BEFORE LLM
    prompt_value = sanitize_text(prompt_value)

    res = invoke_structured(llm, prompt_value, RuleIntegrationResult)

    if debug:
        print("X rule_integration_node")

    return {"tbox": res.tbox}


def code_generation_node(state: State, llm: BaseChatModel, debug: bool):
    if debug:
        print("E code_generation_node")

    if not state.tbox:
        print("err: empty TBox")
        return {}

    # CODE_GENERATION_PROMPT использует template_format="jinja2"
    # и возвращает сырой Python-код, поэтому здесь намеренно llm.invoke()
    prompt_value = CODE_GENERATION_PROMPT.format(
        statute=state.raw_statute_text,
        classes=json.dumps([cls.model_dump() for cls in state.tbox.classes]),
        properties=json.dumps(
            [prop.model_dump() for prop in state.tbox.properties]
            + [
                TBoxProperty(
                    type=rule.implication_property_type,
                    name=rule.implication_property_name,
                    arguments=rule.implication_property_arguments,
                    description=rule.description,
                ).model_dump()
                for rule in state.tbox.rules
            ]
        ),
        rules=json.dumps([rule.model_dump() for rule in state.tbox.rules]),
        last_interpreter=state.tbox_eval_code,
        last_error=(
            "\nTest case that failed:\n"
            f"Description: {state.training_evaluation_error.description}\n"
            f"Question: {state.training_evaluation_error.query}\n\n"
            "For which, given your instructions, the following ABox was constructed:\n"
            f"{state.training_evaluation_error.extracted_abox}\n\n"
            "For which the code failed with the following error:\n"
            f"Error: {state.training_evaluation_error.error}"
        )
        if state.training_evaluation_error
        else None,
    )

    raw = llm.invoke(prompt_value)
    code: str = raw.content if hasattr(raw, "content") else str(raw)

    # 🔥 FIX 5
    code = sanitize_text(code)

    # убираем возможные ```python ... ``` обёртки
    code = re.sub(r"^```(?:python)?\s*", "", code.strip())
    code = re.sub(r"\s*```$", "", code.strip())

    if debug:
        print("X code_generation_node")

    return {"tbox_eval_code": code}


def human_review_node(state: State):
    print(f"\n{'=' * 50}\nREVIEW\n{'=' * 50}")
    if state.tbox:
        issues = validate_tbox(state.tbox)
        if not issues:
            print("\nValid TBox.")
        else:
            print("\nValidation issues:")
            for issue in issues:
                print(f"  - {issue}")

        print("\nClasses:")
        for cls in state.tbox.classes:
            print(f"  - {cls.name}: {cls.description}")

        print("\nProperties:")
        for prop in state.tbox.properties:
            print(f"  - {prop.name} ({prop.type}): {prop.description}")
            print(f"    arguments: {prop.arguments}")

        print("\nRules:")
        for rule in state.tbox.rules:
            print(f"  - {rule.description}")
            print(f"    FOL: {rule.fol_expression}")
            print(
                f"    Implies: {rule.implication_property_name}"
                f"({', '.join(rule.implication_property_arguments)})"
            )

    print("\nOptions:")
    print("1. Approve (type 'approve')")
    print("2. Provide feedback for improvements (type your feedback)")

    while True:
        human_input = input("\nYour response: ").strip()
        if not human_input:
            print("Please provide a valid response.")
            continue
        return {"human_feedback": human_input, "tbox_validation_issues": None}


def training_evaluation_node(
    state: State, llm: BaseChatModel, debug: bool
):
    assert state.tbox is not None
    assert state.tbox_eval_code is not None

    fix_attempts = (
        1 if state.training_evaluation_error is None else state.fix_attempts + 1
    )

    if fix_attempts > MAX_FIX_ATTEMPTS:
        return {"training_evaluation_error": None}

    df = pd.read_csv("./eval/train.csv", sep=";")

    for ix, row in df.iterrows():

        sentence = row["tokens_braced"]
        candidates = row["cands_analtag_line"]

        gold_analysis = row["gold_analysis_line"]
        gold_tags = row["gold_tag_line"]

        try:
            res, abox = knowledge_application_predict(
                llm=llm,
                debug=debug,
                tbox=state.tbox,
                tbox_interpreter=state.tbox_eval_code,
                sentence=sentence,
                uniparser_output=candidates,
            )

        except Exception as ex:
            return {
                "training_evaluation_error": TrainingProblem(
                    description=sentence,
                    query="structured parsing",
                    error=f"{ex}\n{traceback.format_exc()}",
                ),
                "fix_attempts": fix_attempts,
            }

        # res должен содержать 2 строки:
        pred_analysis = res.get("analysis_line")
        pred_tags = res.get("tag_line")

        if pred_analysis != gold_analysis or pred_tags != gold_tags:

            return {
                "training_evaluation_error": TrainingProblem(
                    description=sentence,
                    query="full structured mismatch",
                    extracted_abox=abox,
                    error=(
                        f"Pred analysis: {pred_analysis}\n"
                        f"Gold analysis: {gold_analysis}\n\n"
                        f"Pred tags: {pred_tags}\n"
                        f"Gold tags: {gold_tags}"
                    ),
                ),
                "fix_attempts": fix_attempts,
            }

    return {
        "training_evaluation_error": None,
        "fix_attempts": None,
    }

# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def decide_after_human_review(state: State):
    match (state.human_feedback or "").lower().strip():
        case "approve":
            return END
        case _:
            return "rule_integration"


def tbox_validation_node(state: State):
    if not state.tbox:
        return []

    if state.fix_attempts >= MAX_FIX_ATTEMPTS:
        return {
            "tbox_validation_issues": None,
            "fix_attempts": 0,
            "human_feedback": None,
        }

    return {
        "tbox_validation_issues": validate_tbox(state.tbox),
        "human_feedback": None,
        "fix_attempts": state.fix_attempts + 1,
    }


def decide_after_tbox_validation(state: State):
    if state.tbox_validation_issues:
        return "rule_integration"
    return END


def decide_after_training_evaluation(state: State):
    if state.training_evaluation_error:
        return "code_generation"
    return END


# ---------------------------------------------------------------------------
# Graph construction & entry point
# ---------------------------------------------------------------------------

def graph(llm: BaseChatModel, debug: bool):
    builder = StateGraph(State)

    builder.add_node(
        "concept_extraction", partial(concept_extraction_node, llm=llm, debug=debug)
    )
    builder.add_node(
        "rule_formulation", partial(rule_formulation_node, llm=llm, debug=debug)
    )
    builder.add_node(
        "rule_integration", partial(rule_integration_node, llm=llm, debug=debug)
    )
    builder.add_node("tbox_validation", tbox_validation_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node(
        "code_generation", partial(code_generation_node, llm=llm, debug=debug)
    )
    builder.add_node(
        "training_evaluation",
        partial(training_evaluation_node, llm=llm, debug=debug),
    )

    builder.add_edge(START, "concept_extraction")
    builder.add_edge(START, "rule_formulation")
    builder.add_edge("concept_extraction", "rule_integration")
    builder.add_edge("rule_formulation", "rule_integration")
    builder.add_edge("rule_integration", "tbox_validation")
    builder.add_edge("code_generation", "training_evaluation")

    builder.add_conditional_edges(
        "tbox_validation",
        decide_after_tbox_validation,
        {
            "rule_integration": "rule_integration",
            END: "human_review",
        },
    )
    builder.add_conditional_edges(
        "human_review",
        decide_after_human_review,
        {
            "rule_integration": "rule_integration",
            END: "code_generation",
        },
    )
    builder.add_conditional_edges(
        "training_evaluation",
        decide_after_training_evaluation,
        {
            "code_generation": "code_generation",
            END: END,
        },
    )

    return builder.compile()


def predict(llm: BaseChatModel, input: str, debug: bool) -> tuple[TBox, str]:
    app = graph(llm, debug)
    if debug:
        print(app.get_graph().draw_ascii())

    final_state = app.invoke(State(raw_statute_text=input))
    tbox = final_state["tbox"]
    if debug:
        print(json.dumps(tbox.model_dump(), indent=2))
        print("\n\nTBox Evaluation Code\n\n")
        print(final_state["tbox_eval_code"])
    return tbox, final_state["tbox_eval_code"]