#!/usr/bin/env python3
"""Prototype pyformlang translation for the O5 simple visible grammar draft."""

from __future__ import annotations

from dataclasses import dataclass

from pyformlang.cfg import CFG, Production, Terminal, Variable


TOK = {
    "UNIT_START": "<unit>",
    "UNIT_END": "</unit>",
    "NON_INFER": "NON_INFER",
    "SPOKEN": "SPOKEN",
    "THINK_START": "<think>",
    "THINK_END": "</think>",
    "TOOL_CALL_START": "<tool-call>",
    "TOOL_CALL_END": "</tool-call>",
    "TOOL_ABORT": "<tool-abort>",
    "TOOL_RESPONSE_START": "<tool-response>",
    "TOOL_RESPONSE_END": "</tool-response>",
    "BUDGET_REACHED": "<budget-reached>",
    "THINK_TOKEN": "THINK_TOKEN",
    "TOOL_CALL_TOKEN": "TOOL_CALL_TOKEN",
    "TOOL_RESPONSE_JSON_ID_RESULT": "TOOL_RESPONSE_JSON_ID_RESULT",
}

IGNORE = {TOK["BUDGET_REACHED"]}


@dataclass(frozen=True)
class Unit:
    content: tuple[str, ...]


def t(name: str) -> Terminal:
    return Terminal(TOK[name])


def v(name: str) -> Variable:
    return Variable(name)


def build_infer_cfg() -> CFG:
    """Builds INFER_STREAM after erase_ignore(INFER_PROJECTION(...))."""

    s = v("INFER_STREAM")
    top_items = v("TOP_ITEMS")
    top_item = v("TOP_ITEM")
    think_block = v("THINK_BLOCK")
    think_body = v("THINK_BODY")
    tool_call_block = v("TOOL_CALL_BLOCK")
    tool_call_body = v("TOOL_CALL_BODY")
    tool_response_block = v("TOOL_RESPONSE_BLOCK")

    productions = {
        Production(s, [top_items]),
        Production(top_items, [top_item]),
        Production(top_items, [top_item, top_items]),
        Production(top_item, [think_block]),
        Production(top_item, [tool_call_block]),
        Production(top_item, [tool_response_block]),
        Production(
            think_block,
            [t("THINK_START"), think_body, t("THINK_END")],
        ),
        Production(think_body, []),
        Production(think_body, [t("THINK_TOKEN"), think_body]),
        Production(
            tool_call_block,
            [t("TOOL_CALL_START"), tool_call_body, t("TOOL_CALL_END")],
        ),
        Production(
            tool_call_block,
            [t("TOOL_CALL_START"), tool_call_body, t("TOOL_ABORT")],
        ),
        Production(tool_call_body, []),
        Production(tool_call_body, [t("TOOL_CALL_TOKEN"), tool_call_body]),
        Production(
            tool_response_block,
            [
                t("TOOL_RESPONSE_START"),
                t("TOOL_RESPONSE_JSON_ID_RESULT"),
                t("TOOL_RESPONSE_END"),
            ],
        ),
    }

    return CFG(start_symbol=s, productions=productions)


def parse_units(tokens: list[str]) -> list[Unit]:
    """Parses SESSION := UNIT+ with UNIT_CONTENT := INFER_TOKEN+."""

    units: list[Unit] = []
    i = 0
    n = len(tokens)
    infer_tokens = {
        TOK["THINK_START"],
        TOK["THINK_END"],
        TOK["TOOL_CALL_START"],
        TOK["TOOL_CALL_END"],
        TOK["TOOL_ABORT"],
        TOK["TOOL_RESPONSE_START"],
        TOK["TOOL_RESPONSE_END"],
        TOK["BUDGET_REACHED"],
        TOK["THINK_TOKEN"],
        TOK["TOOL_CALL_TOKEN"],
        TOK["TOOL_RESPONSE_JSON_ID_RESULT"],
    }

    while i < n:
        if tokens[i : i + 3] != [TOK["UNIT_START"], TOK["NON_INFER"], TOK["SPOKEN"]]:
            raise ValueError(f"invalid unit prefix at token {i}")
        i += 3

        content: list[str] = []
        while i < n and tokens[i] != TOK["UNIT_END"]:
            if tokens[i] not in infer_tokens:
                raise ValueError(f"invalid infer token at token {i}: {tokens[i]}")
            content.append(tokens[i])
            i += 1

        if not content:
            raise ValueError("unit content must be non-empty")
        if i >= n or tokens[i] != TOK["UNIT_END"]:
            raise ValueError("unterminated unit")
        i += 1
        units.append(Unit(tuple(content)))

    if not units:
        raise ValueError("session must contain at least one unit")
    return units


def infer_projection(units: list[Unit]) -> list[str]:
    return [tok for unit in units for tok in unit.content]


def erase_ignore(tokens: list[str]) -> list[str]:
    return [tok for tok in tokens if tok not in IGNORE]


def is_valid_session(tokens: list[str], infer_cfg: CFG) -> bool:
    try:
        units = parse_units(tokens)
    except ValueError:
        return False
    normalized = erase_ignore(infer_projection(units))
    return infer_cfg.contains(normalized)


def main() -> None:
    cfg = build_infer_cfg()

    cases = {
        "think_complete": [
            TOK["UNIT_START"],
            TOK["NON_INFER"],
            TOK["SPOKEN"],
            TOK["THINK_START"],
            TOK["THINK_TOKEN"],
            TOK["THINK_END"],
            TOK["UNIT_END"],
        ],
        "think_cross_unit": [
            TOK["UNIT_START"],
            TOK["NON_INFER"],
            TOK["SPOKEN"],
            TOK["THINK_START"],
            TOK["THINK_TOKEN"],
            TOK["UNIT_END"],
            TOK["UNIT_START"],
            TOK["NON_INFER"],
            TOK["SPOKEN"],
            TOK["THINK_TOKEN"],
            TOK["THINK_END"],
            TOK["UNIT_END"],
        ],
        "budget_ignored": [
            TOK["UNIT_START"],
            TOK["NON_INFER"],
            TOK["SPOKEN"],
            TOK["BUDGET_REACHED"],
            TOK["THINK_START"],
            TOK["BUDGET_REACHED"],
            TOK["THINK_END"],
            TOK["UNIT_END"],
        ],
        "budget_only_invalid": [
            TOK["UNIT_START"],
            TOK["NON_INFER"],
            TOK["SPOKEN"],
            TOK["BUDGET_REACHED"],
            TOK["UNIT_END"],
        ],
        "empty_unit_content": [
            TOK["UNIT_START"],
            TOK["NON_INFER"],
            TOK["SPOKEN"],
            TOK["UNIT_END"],
        ],
        "overlap_invalid": [
            TOK["UNIT_START"],
            TOK["NON_INFER"],
            TOK["SPOKEN"],
            TOK["THINK_START"],
            TOK["TOOL_CALL_START"],
            TOK["TOOL_CALL_END"],
            TOK["THINK_END"],
            TOK["UNIT_END"],
        ],
    }

    for name, tokens in cases.items():
        print(f"{name}: {is_valid_session(tokens, cfg)}")


if __name__ == "__main__":
    main()
