from functools import partial
from typing import Callable

from langchain.chat_models.base import BaseChatModel
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from .model import (
    ABox,
    TBox,
)
from .prompts import FACT_EXTRACTION_PROMPT


# =========================================================
# STATE
# =========================================================

class State(BaseModel):
    sentence: str
    uniparser_output: str

    tbox: TBox
    tbox_callable: Callable[..., object]

    abox: ABox | None = None

    result: dict | None = None


# =========================================================
# FACT EXTRACTION RESULT
# =========================================================

class FactExtractionResult(BaseModel):
    abox: ABox = Field(...)


# =========================================================
# FACT EXTRACTION NODE
# =========================================================

def fact_extraction_node(
    state: State,
    llm: BaseChatModel,
    debug: bool,
):
    if debug:
        print("E fact_extraction_node")

    properties_str = ""

    for p in state.tbox.properties:

        if len(p.arguments) == 1:
            properties_str += (
                f"- {p.name}({p.arguments[0]}): {p.description}\n"
            )

        elif len(p.arguments) >= 2:
            properties_str += (
                f"- {p.name}({p.arguments[0]}, {p.arguments[1]}): "
                f"{p.description}\n"
            )

    prompt = FACT_EXTRACTION_PROMPT.format(
        sentence=state.sentence,
        uniparser_output=state.uniparser_output,
        classes="\n".join(
            [f"- {c.name}: {c.description}" for c in state.tbox.classes]
        ),
        properties=properties_str,
        tbox_interpreter_docstring=state.tbox_callable.__doc__,
    )

    res = llm.with_structured_output(
        FactExtractionResult
    ).invoke(prompt)

    if debug:
        print("X fact_extraction_node")

    return {
        "abox": res.abox
    }


# =========================================================
# INTERPRETER NODE
# =========================================================

def interpreter_node(
    state: State,
    debug: bool,
):

    if debug:
        print("E interpreter_node")

    if not state.abox:
        raise Exception("ABox missing")

    abox_dict = state.abox.as_dict()

    # ищем sentence individual
    sent_ids = []

    for ind in abox_dict["individuals"]:

        ind_data = abox_dict["individuals"][ind]

        if ind_data["type"] == "Sentence":
            sent_ids.append(ind)

    if not sent_ids:
        raise Exception("Sentence individual not found")

    sent_id = sent_ids[0]

    result = state.tbox_callable(
        abox_dict,
        sent_id,
    )

    if debug:
        print("X interpreter_node")

    return {
        "result": result
    }


# =========================================================
# GRAPH
# =========================================================

def graph(
    llm: BaseChatModel,
    debug: bool,
):

    builder = StateGraph(State)

    builder.add_node(
        "fact_extraction",
        partial(
            fact_extraction_node,
            llm=llm,
            debug=debug,
        )
    )

    builder.add_node(
        "interpreter",
        partial(
            interpreter_node,
            debug=debug,
        )
    )

    builder.add_edge(
        START,
        "fact_extraction"
    )

    builder.add_edge(
        "fact_extraction",
        "interpreter"
    )

    builder.add_edge(
        "interpreter",
        END
    )

    return builder.compile()


# =========================================================
# ENTRY POINT
# =========================================================

def predict(
    llm: BaseChatModel,
    sentence: str,
    uniparser_output: str,
    tbox: TBox,
    tbox_interpreter: str,
    debug: bool,
):

    app = graph(llm, debug)

    local_namespace = {}
    exec(tbox_interpreter, local_namespace, local_namespace)

    disambiguate = local_namespace["disambiguate"]
    calculate = local_namespace["calculate"]

    tbox_callable = local_namespace.get("calculate")

    if not callable(tbox_callable):
        raise Exception(
            "'calculate' function not found"
        )

    final_state = app.invoke(
        State(
            sentence=sentence,
            uniparser_output=uniparser_output,
            tbox=tbox,
            tbox_callable=tbox_callable,
        )
    )

    return (
        final_state["result"],
        final_state["abox"].as_dict(),
    )