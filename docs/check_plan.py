"""Check that docs/todo.md is a consistent plan.

Checks that every task's "Depends on" line, the mermaid graph and the dependency
table agree; that there are no cycles, orphan tasks, duplicate or dangling
subtask ids; and that headings match task titles.

Run from the repo root: ``python docs/check_plan.py``. Exits 1 on any problem.
"""

import re
import sys
from collections import Counter
from pathlib import Path

PLAN = Path(__file__).parent / "todo.md"
TASK = re.compile(
    r"^- \[[ x]\] \*\*(T\d+)\. (.*?)\*\* — Depends on: (.*?) · PRD:", re.M
)
SUBTASK = re.compile(r"^  - \[[ x]\] (T\d+\.\d+\w?) ", re.M)
TABLE_ROW = re.compile(r"^\| (T\d+) \| (.*?) \| (.*?) \|$", re.M)
FINAL_TASK = "T39"


def _ids(text: str) -> set[str]:
    return set(re.findall(r"T\d+", text))


def _subtask_key(sid: str) -> tuple[int, int, str]:
    m = re.match(r"T(\d+)\.(\d+)(\w?)", sid)
    assert m is not None
    return int(m.group(1)), int(m.group(2)), m.group(3)


def check(text: str) -> list[str]:
    """Return a list of problems found in the plan text."""
    problems: list[str] = []
    deps = {m.group(1): _ids(m.group(3)) for m in TASK.finditer(text)}
    titles = {m.group(1): m.group(2) for m in TASK.finditer(text)}

    graph: dict[str, set[str]] = {t: set() for t in deps}
    mermaid = text.split("```mermaid")[1].split("```")[0]
    for line in mermaid.splitlines():
        ids = re.findall(r"T\d+", line)
        for a, b in zip(ids, ids[1:]):
            graph[b].add(a)

    table = {
        m.group(1): (_ids(m.group(2)), _ids(m.group(3)))
        for m in TABLE_ROW.finditer(text)
    }
    headings = dict(re.findall(r"^### (T\d+)\. (.*)$", text, re.M))
    for task, needs in deps.items():
        unlocks = {u for u in deps if task in deps[u]}
        if graph[task] != needs:
            problems.append(f"{task}: mermaid graph disagrees with 'Depends on'")
        if table.get(task) != (needs, unlocks):
            problems.append(f"{task}: dependency table disagrees with 'Depends on'")
        if not unlocks and task != FINAL_TASK:
            problems.append(f"{task}: orphan (unlocks no task)")
        if headings.get(task) != titles[task]:
            problems.append(f"{task}: heading does not match the task title")

    state: dict[str, int] = {}

    def visit(task: str, path: list[str]) -> None:
        if state.get(task) == 1:
            problems.append(f"cycle: {' -> '.join(path + [task])}")
            return
        if state.get(task) == 2:
            return
        state[task] = 1
        for needed in deps[task]:
            visit(needed, path + [task])
        state[task] = 2

    for task in deps:
        visit(task, [])

    subtasks = SUBTASK.findall(text)
    problems += [
        f"{s}: duplicate subtask id" for s, n in Counter(subtasks).items() if n > 1
    ]
    for a, b in zip(subtasks, subtasks[1:]):
        key_a, key_b = _subtask_key(a), _subtask_key(b)
        if key_a[0] == key_b[0] and key_a >= key_b:
            problems.append(f"{a}, {b}: subtasks out of order")
    known = set(subtasks)
    for ref in sorted(set(re.findall(r"T\d+\.\d+\w?", text))):
        if ref not in known and ref.rstrip("ab") not in known:
            problems.append(f"{ref}: referenced but not defined")
    return problems


def main() -> int:
    """Print the plan summary and any problems; return the exit code."""
    text = PLAN.read_text()
    problems = check(text)
    print(
        f"{len(TASK.findall(text))} tasks, {len(SUBTASK.findall(text))} subtasks, "
        f"{len(problems)} problems"
    )
    for problem in problems:
        print(f"  {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
