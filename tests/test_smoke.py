"""Stage-0 smoke tests: the package must be importable and its metadata sane."""

import agent_base


def test_package_importable() -> None:
    assert agent_base.__version__


def test_version_is_semver() -> None:
    parts = agent_base.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts), agent_base.__version__
