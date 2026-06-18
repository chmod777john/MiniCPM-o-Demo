#!/usr/bin/env python3
"""Prototype Pynini/OpenFST operations for the O5 simple grammar draft."""

from __future__ import annotations

import pynini


TOKENS = [
    "UNIT_START",
    "UNIT_END",
    "NON_INFER",
    "SPOKEN",
    "THINK_START",
    "THINK_END",
    "TOOL_CALL_START",
    "TOOL_CALL_END",
    "TOOL_ABORT",
    "TOOL_RESPONSE_START",
    "TOOL_RESPONSE_END",
    "BUDGET_REACHED",
    "THINK_TOKEN",
    "TOOL_CALL_TOKEN",
    "TOOL_RESPONSE_JSON_ID_RESULT",
]

SYM = {name: chr(ord("A") + i) for i, name in enumerate(TOKENS)}
NAME = {ord(value): name for name, value in SYM.items()}


def s(*names: str) -> str:
    return "".join(SYM[name] for name in names)


def accep(*names: str) -> pynini.Fst:
    return pynini.accep(s(*names))


def union(*fsts: pynini.Fst) -> pynini.Fst:
    return pynini.union(*fsts).optimize()


def star(fst: pynini.Fst) -> pynini.Fst:
    return pynini.closure(fst).optimize()


def plus(fst: pynini.Fst) -> pynini.Fst:
    return pynini.closure(fst, 1).optimize()


def concat(*fsts: pynini.Fst) -> pynini.Fst:
    out = pynini.accep("")
    for fst in fsts:
        out = out + fst
    return out.optimize()


def is_nonempty(fst: pynini.Fst) -> bool:
    return fst.start() != pynini.NO_STATE_ID


def accepts(acceptor: pynini.Fst, tokens: list[str]) -> bool:
    return is_nonempty(pynini.accep(s(*tokens)) @ acceptor)


def build_visible_dfa() -> pynini.Fst:
    infer_token = union(
        accep("THINK_START"),
        accep("THINK_END"),
        accep("TOOL_CALL_START"),
        accep("TOOL_CALL_END"),
        accep("TOOL_ABORT"),
        accep("TOOL_RESPONSE_START"),
        accep("TOOL_RESPONSE_END"),
        accep("BUDGET_REACHED"),
        accep("THINK_TOKEN"),
        accep("TOOL_CALL_TOKEN"),
        accep("TOOL_RESPONSE_JSON_ID_RESULT"),
    )
    unit_prefix = concat(accep("UNIT_START"), accep("NON_INFER"), accep("SPOKEN"))
    unit = concat(unit_prefix, plus(infer_token), accep("UNIT_END"))
    return plus(unit).optimize()


def build_infer_dfa() -> pynini.Fst:
    think_body = star(accep("THINK_TOKEN"))
    think_block = concat(accep("THINK_START"), think_body, accep("THINK_END"))

    tool_body = star(accep("TOOL_CALL_TOKEN"))
    tool_call_block = union(
        concat(accep("TOOL_CALL_START"), tool_body, accep("TOOL_CALL_END")),
        concat(accep("TOOL_CALL_START"), tool_body, accep("TOOL_ABORT")),
    )

    tool_response_block = concat(
        accep("TOOL_RESPONSE_START"),
        accep("TOOL_RESPONSE_JSON_ID_RESULT"),
        accep("TOOL_RESPONSE_END"),
    )

    top_item = union(think_block, tool_call_block, tool_response_block)
    return plus(top_item).optimize()


def build_projection_fst() -> pynini.Fst:
    """Maps visible SESSION to INFER_PROJECTION by deleting unit prefix/end."""

    deletes = union(
        pynini.cross(SYM["UNIT_START"], ""),
        pynini.cross(SYM["UNIT_END"], ""),
        pynini.cross(SYM["NON_INFER"], ""),
        pynini.cross(SYM["SPOKEN"], ""),
    )
    keeps = union(
        *[
            pynini.cross(SYM[name], SYM[name])
            for name in TOKENS
            if name not in {"UNIT_START", "UNIT_END", "NON_INFER", "SPOKEN"}
        ]
    )
    return star(union(deletes, keeps)).optimize()


def build_ignore_erase_fst() -> pynini.Fst:
    """Maps INFER_PROJECTION to INFER_NORMALIZATION by deleting budget tokens."""

    deletes = pynini.cross(SYM["BUDGET_REACHED"], "")
    keeps = union(
        *[
            pynini.cross(SYM[name], SYM[name])
            for name in TOKENS
            if name
            not in {
                "UNIT_START",
                "UNIT_END",
                "NON_INFER",
                "SPOKEN",
                "BUDGET_REACHED",
            }
        ]
    )
    return star(union(deletes, keeps)).optimize()


def valid_by_fst(tokens: list[str]) -> bool:
    expanded = build_expanded_visible_dfa()

    visible_input = pynini.accep(s(*tokens))
    return is_nonempty(visible_input @ expanded)


def build_expanded_visible_dfa() -> pynini.Fst:
    """Compiles projection/ignore/infer constraints into one visible acceptor."""

    visible_dfa = build_visible_dfa()
    infer_dfa = build_infer_dfa()
    projection = build_projection_fst()
    erase_ignore = build_ignore_erase_fst()

    valid_via_projection = projection @ erase_ignore @ infer_dfa
    valid_visible_inputs = pynini.project(valid_via_projection, "input").optimize()
    expanded = (visible_dfa @ valid_visible_inputs).optimize()
    expanded = pynini.determinize(expanded).optimize()
    return pynini.minimize(expanded).optimize()


def print_state_table(fst: pynini.Fst) -> None:
    print(f"expanded_states: {fst.num_states()}")
    print(f"expanded_start: {fst.start()}")
    finals = [state for state in fst.states() if fst.final(state) != pynini.Weight.zero(fst.weight_type())]
    print(f"expanded_finals: {finals}")
    for state in fst.states():
        for arc in fst.arcs(state):
            ilabel = "ε" if arc.ilabel == 0 else NAME.get(arc.ilabel, f"#{arc.ilabel}")
            olabel = "ε" if arc.olabel == 0 else NAME.get(arc.olabel, f"#{arc.olabel}")
            if ilabel == olabel:
                label = ilabel
            else:
                label = f"{ilabel}:{olabel}"
            print(f"{state} -- {label} --> {arc.nextstate}")


def decode_symbols(encoded: str) -> list[str]:
    return [NAME.get(ord(ch), f"#{ord(ch)}") for ch in encoded]


def sample_accepted_sequences(fst: pynini.Fst, n: int = 20) -> list[list[str]]:
    paths_fst = pynini.shortestpath(fst, nshortest=n, unique=True).optimize()
    paths = paths_fst.paths()
    samples: list[list[str]] = []
    while not paths.done():
        samples.append(decode_symbols(paths.istring()))
        paths.next()
    return samples


def shortest_example(fst: pynini.Fst, constraint: pynini.Fst) -> list[str] | None:
    candidates = (fst @ constraint).optimize()
    if not is_nonempty(candidates):
        return None
    best = pynini.shortestpath(candidates, nshortest=1, unique=True).optimize()
    paths = best.paths()
    if paths.done():
        return None
    return decode_symbols(paths.istring())


def synthesize_named_examples(fst: pynini.Fst) -> list[tuple[str, list[str]]]:
    def exact(*names: str) -> pynini.Fst:
        return accep(*names)

    scenarios = [
        (
            "think_empty_body",
            exact(
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "THINK_START",
                "THINK_END",
                "UNIT_END",
            ),
        ),
        (
            "think_with_body",
            exact(
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "THINK_START",
                "THINK_TOKEN",
                "THINK_END",
                "UNIT_END",
            ),
        ),
        (
            "think_cross_unit",
            exact(
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "THINK_START",
                "THINK_TOKEN",
                "UNIT_END",
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "THINK_END",
                "UNIT_END",
            ),
        ),
        (
            "tool_call_end",
            exact(
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "TOOL_CALL_START",
                "TOOL_CALL_END",
                "UNIT_END",
            ),
        ),
        (
            "tool_call_abort",
            exact(
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "TOOL_CALL_START",
                "TOOL_ABORT",
                "UNIT_END",
            ),
        ),
        (
            "tool_response",
            exact(
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "TOOL_RESPONSE_START",
                "TOOL_RESPONSE_JSON_ID_RESULT",
                "TOOL_RESPONSE_END",
                "UNIT_END",
            ),
        ),
        (
            "budget_ignored",
            exact(
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "BUDGET_REACHED",
                "THINK_START",
                "BUDGET_REACHED",
                "THINK_END",
                "UNIT_END",
            ),
        ),
        (
            "two_items_one_unit",
            exact(
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "THINK_START",
                "THINK_END",
                "TOOL_RESPONSE_START",
                "TOOL_RESPONSE_JSON_ID_RESULT",
                "TOOL_RESPONSE_END",
                "UNIT_END",
            ),
        ),
        (
            "tool_call_cross_unit",
            exact(
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "TOOL_CALL_START",
                "UNIT_END",
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "TOOL_CALL_END",
                "UNIT_END",
            ),
        ),
        (
            "tool_response_cross_unit",
            exact(
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "TOOL_RESPONSE_START",
                "TOOL_RESPONSE_JSON_ID_RESULT",
                "UNIT_END",
                "UNIT_START",
                "NON_INFER",
                "SPOKEN",
                "TOOL_RESPONSE_END",
                "UNIT_END",
            ),
        ),
    ]

    examples: list[tuple[str, list[str]]] = []
    for name, constraint in scenarios:
        example = shortest_example(fst, constraint)
        if example is not None:
            examples.append((name, example))
    return examples


def state_aliases(fst: pynini.Fst) -> dict[int, str]:
    aliases = {state: f"S{state}" for state in fst.states()}
    special = {
        0: "START",
        1: "SAW_UNIT_START",
        2: "SAW_NON_INFER",
        3: "READY",
        4: "IN_THINK",
        5: "IN_TOOL_CALL",
        6: "IN_TOOL_RESPONSE",
        7: "READY_AFTER_IGNORE_ONLY",
        8: "THINK_WAIT_NEXT_UNIT",
        9: "AFTER_ITEM",
        10: "TOOL_CALL_WAIT_NEXT_UNIT",
        11: "TOOL_RESPONSE_WAIT_NEXT_UNIT",
        12: "IN_TOOL_RESPONSE_AFTER_JSON",
        14: "FINAL_AFTER_UNIT",
    }
    for state, alias in special.items():
        if state < fst.num_states():
            aliases[state] = alias
    return aliases


def write_state_table_markdown(fst: pynini.Fst, path: str) -> None:
    aliases = state_aliases(fst)
    finals = {state for state in fst.states() if fst.final(state) != pynini.Weight.zero(fst.weight_type())}
    lines = [
        "# O5 最简协议展开状态机",
        "",
        "此文件由 `agent-harness/scripts/o5_simple_grammar_pynini.py` 生成。",
        "",
        "## 状态",
        "",
        "| 状态 | 含义 |",
        "|---|---|",
    ]
    for state in fst.states():
        marks = []
        if state == fst.start():
            marks.append("start")
        if state in finals:
            marks.append("final")
        suffix = f" ({', '.join(marks)})" if marks else ""
        lines.append(f"| `{aliases[state]}` | state {state}{suffix} |")

    lines.extend(
        [
            "",
            "## 转移",
            "",
            "| From | Token | To |",
            "|---|---|---|",
        ]
    )
    for state in fst.states():
        for arc in fst.arcs(state):
            ilabel = "ε" if arc.ilabel == 0 else NAME.get(arc.ilabel, f"#{arc.ilabel}")
            olabel = "ε" if arc.olabel == 0 else NAME.get(arc.olabel, f"#{arc.olabel}")
            label = ilabel if ilabel == olabel else f"{ilabel}:{olabel}"
            lines.append(f"| `{aliases[state]}` | `{label}` | `{aliases[arc.nextstate]}` |")

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_examples_markdown(samples: list[list[str]], path: str) -> None:
    lines = [
        "# O5 最简协议自动生成样例",
        "",
        "此文件由 `agent-harness/scripts/o5_simple_grammar_pynini.py` 从展开 DFA 的 accepted paths 生成。",
        "",
    ]
    for idx, sample in enumerate(samples, start=1):
        lines.append(f"## Example {idx}")
        lines.append("")
        lines.append("```text")
        lines.extend(sample)
        lines.append("```")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_named_examples_markdown(examples: list[tuple[str, list[str]]], path: str) -> None:
    lines = [
        "# O5 最简协议场景样例",
        "",
        "此文件由 `agent-harness/scripts/o5_simple_grammar_pynini.py` 生成。",
        "",
        "每个样例由一个场景约束与展开后的 visible DFA 求交，再取最短 accepted path 得到。",
        "",
    ]
    for name, sample in examples:
        lines.append(f"## {name}")
        lines.append("")
        lines.append("```text")
        lines.extend(sample)
        lines.append("```")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    cases = {
        "think_complete": [
            "UNIT_START",
            "NON_INFER",
            "SPOKEN",
            "THINK_START",
            "THINK_TOKEN",
            "THINK_END",
            "UNIT_END",
        ],
        "think_cross_unit": [
            "UNIT_START",
            "NON_INFER",
            "SPOKEN",
            "THINK_START",
            "THINK_TOKEN",
            "UNIT_END",
            "UNIT_START",
            "NON_INFER",
            "SPOKEN",
            "THINK_TOKEN",
            "THINK_END",
            "UNIT_END",
        ],
        "budget_ignored": [
            "UNIT_START",
            "NON_INFER",
            "SPOKEN",
            "BUDGET_REACHED",
            "THINK_START",
            "BUDGET_REACHED",
            "THINK_END",
            "UNIT_END",
        ],
        "budget_only_invalid": [
            "UNIT_START",
            "NON_INFER",
            "SPOKEN",
            "BUDGET_REACHED",
            "UNIT_END",
        ],
        "empty_unit_content": [
            "UNIT_START",
            "NON_INFER",
            "SPOKEN",
            "UNIT_END",
        ],
        "overlap_invalid": [
            "UNIT_START",
            "NON_INFER",
            "SPOKEN",
            "THINK_START",
            "TOOL_CALL_START",
            "TOOL_CALL_END",
            "THINK_END",
            "UNIT_END",
        ],
    }

    for name, tokens in cases.items():
        print(f"{name}: {valid_by_fst(tokens)}")

    print()
    expanded = build_expanded_visible_dfa()
    print_state_table(expanded)
    write_state_table_markdown(
        expanded,
        "agent-harness/o5-simple-expanded-visible-dfa.md",
    )
    write_examples_markdown(
        sample_accepted_sequences(expanded, 20),
        "agent-harness/o5-simple-generated-examples.md",
    )
    write_named_examples_markdown(
        synthesize_named_examples(expanded),
        "agent-harness/o5-simple-scenario-examples.md",
    )


if __name__ == "__main__":
    main()
