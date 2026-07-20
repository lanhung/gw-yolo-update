from __future__ import annotations

import os
import platform
import shlex
import sys
from typing import Any

import numpy as np


def execution_provenance(torch_module: Any | None = None) -> dict[str, Any]:
    """Return the minimum execution identity required by repository reports."""
    environment: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if torch_module is not None:
        cuda_available = bool(torch_module.cuda.is_available())
        environment.update(
            {
                "torch": torch_module.__version__,
                "cuda_available": cuda_available,
                "cuda_version": torch_module.version.cuda,
                "gpu": torch_module.cuda.get_device_name(0) if cuda_available else None,
            }
        )
    return {
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
        "environment": environment,
    }
