# Python Coding & Editing Guidelines

> **Living document – PRs welcome!**
> Last updated: 2025‑07‑15

## Table of Contents

1. Philosophy
1. Docstrings & Comments
1. Type Hints
1. Documentation

---

## Philosophy

- **Readability, reproducibility, performance – in that order.**
- Prefer explicit over implicit; avoid hidden state and global flags.
- Measure before you optimize (`time.perf_counter`, `line_profiler`).
- Each module holds a **single responsibility**; keep public APIs minimal.

## Docstrings & Comments

- Style: NumPyDoc.
- Start with a one‑sentence summary in the imperative mood.
- Sections: Parameters, Returns, Raises, Examples, References.
- Use backticks for code or referring to variables (e.g. `xarray.DataArray`).
- Do not use emojis, or non-unicode characters in comments/print statements.
- Cite peer‑reviewed papers with DOI links when relevant.
- Write code that explains itself rather than needs comments.
- For the inline you do add, explain *why*, not what. For example, *don't* write:

```python
# open the file
f = open(filename)
```

- The comments should be things which are not obvious to a reader with typical background knowledge.

## Tools

- ruff is use for most code maintenance, black for formatting, mypy for type checking, pytest for testing
- You can run `pre-commit run -a` to run all pre-commit hooks and check for style violations

## Code Style

- Annotate all public functions (PEP 484).
- Prefer `Protocol` over `ABC`s when only an interface is needed.
- Validate external inputs via Pydantic models (if existing); otherwise, use `dataclasses`
- Parse, don't validate, with your dataclasses. Checks should be at the serialization boundaries, not scattered everywhere in the code.
- If you need to add an ignore, ignore a specific check like # type: ignore[specific]
- Don't write error handing code or smooth over exceptions/errors unless they are expected as part of control flow.
- In general, write code that will raise an exception early if something isn't expected.
- Enforce important expectations with asserts, but raise errors for user-facing problems.

## Documentation

- mkdocs + Jupyter. Hosted on ReadTheDocs.
- Auto API from type hints.
- Provide tutorial notebooks covering common workflows.
- Include examples in docstrings.
- Add high-level guides for key functionality.

## Project Index (cyclical indexing)

`docs/INDEX.md` is the map of this project: which documents exist, which module
implements which requirement, which task owns it, and what state it is in. It is
re-built on a fixed cycle, not when someone remembers.

**Read `docs/INDEX.md` first** at the start of every session, before opening
`docs/specs.md`, `docs/todo.md` or the source. If its "Last indexed" commit is
not `HEAD`, or files changed since, run an indexing cycle before doing other work.

### When to run a cycle

| Trigger | Scope |
| --- | --- |
| Start of a session, index is stale (commit differs, or untracked/modified files it does not list) | Full |
| A todo task or subtask is finished | The rows for that task, its requirement and its modules |
| A module is added, moved, renamed or deleted | Module map |
| `specs.md` changes (new requirement, decision, open question closed) | Requirement map, decisions log, then `todo.md` |
| A milestone is released (T8, T18, T34, T39) | Full, plus an audit of the three docs against each other |
| A branch is merged | Full |

### What one cycle does

1. **Scan.** List the tracked and untracked files under `src/`, `tests/`, `docs/`
   and `notebooks/` (`git ls-files`, `git status --short`) and the public names in
   `disp_xr/__init__.py`.
2. **Diff.** Compare with the previous index: added, removed, moved, changed.
3. **Re-link.** For every module, set the requirement ids it implements
   (`F*`, `S*`, `C*` in `specs.md`) and the owning task ids (`T*` in `todo.md`).
   For every requirement, set its tasks and status. Every requirement must have
   a task and every module must map to a requirement or be marked infrastructure.
4. **Sync the docs.** Tick finished checkboxes in `todo.md`. Move decisions out of
   "Open questions" in `specs.md` into the decisions log. Fix any statement in
   `specs.md`, `todo.md` or `code_review.md` that the change made false. A change
   to a task's dependencies is made in three places in `todo.md`: its
   "Depends on" line, the mermaid graph and the dependency table.
5. **Check.** No orphan tasks, no dependency cycles, no duplicate ids, no
   requirement without a task, no link to a file that does not exist.
   `python docs/check_plan.py` checks the plan mechanically (dependency lines,
   graph and table agree; no cycles, orphans, duplicate or dangling ids).
6. **Stamp.** Update "Last indexed" (date, commit) and add one line to the index
   log saying what triggered the cycle and what changed.

### Rules

- The index describes what **is**, verified from the files in this cycle. Never
  copy a status forward from the previous index without checking it.
- Things that are broken, stale or contradictory are listed in the index under
  "Out of sync", not left out.
- `specs.md` says what to build, `todo.md` says in what order, `code_review.md`
  says what exists, `porting.md` says what is ported from opera_utils, and
  `references.md` says which outside practices are followed and why. The index only points at them; it does not repeat their content.
- An indexing cycle is part of the task that triggered it, and goes in the same commit.
