from __future__ import annotations

import tempfile
from pathlib import Path


def pytest_configure(config) -> None:
    """Keep pytest temp files outside the repository.

    ``tmp_path`` under the source tree is unreliable on Windows when antivirus
    or an editor holds a handle to ``.pytest_tmp``.  A unique directory in the
    OS temp area avoids that failure without affecting test isolation.
    """

    if not config.option.basetemp:
        config.option.basetemp = Path(tempfile.mkdtemp(prefix="web2spec-pytest-"))
