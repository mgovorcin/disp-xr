from pathlib import Path

import pytest
from fixtures.make_disp import DispStack, make_disp_stack


class _Stacks(dict):
    """Write each requested variant once, on first access."""

    def __init__(self, base: Path):
        super().__init__()
        self.base = base

    def __missing__(self, variant: str) -> DispStack:
        self[variant] = make_disp_stack(self.base / variant, variant)
        return self[variant]


@pytest.fixture(scope="session")
def disp_stacks(tmp_path_factory) -> _Stacks:
    """Synthetic stacks by variant name, shared across the session (read only)."""
    return _Stacks(tmp_path_factory.mktemp("disp"))


@pytest.fixture
def two_ministacks(disp_stacks) -> DispStack:
    return disp_stacks["two_ministacks"]


@pytest.fixture
def stack_dir(two_ministacks) -> Path:
    return two_ministacks.paths[0].parent
