import pytest

from prep.api import ProcArgs
from prep.constants import is_env_enabled


@pytest.fixture(scope="session", autouse=True)
def procargs():
    return ProcArgs(num_proc=1)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Modify test collection based on markers and environment."""
    _handle_slow_tests(items)


def _handle_slow_tests(items: list[pytest.Item]):
    """Skip slow tests unless RUN_SLOW is enabled."""
    if is_env_enabled("RUN_SLOW"):
        return

    skip_slow = pytest.mark.skip(reason="slow test (set RUN_SLOW=1 to run)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
