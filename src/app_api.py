from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


SERVICE_PATH = (
    Path(__file__).resolve().parent
    / "19_fastapi_service.py"
)


def load_fastapi_service() -> ModuleType:
    if not SERVICE_PATH.is_file():
        raise FileNotFoundError(
            "HeveMind FastAPI service was not found at: "
            f"{SERVICE_PATH}"
        )

    module_name = "hevemind_fastapi_service"

    spec = importlib.util.spec_from_file_location(
        module_name,
        SERVICE_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to create an import specification for: "
            f"{SERVICE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)

    # Critical: register the module before execution so Pydantic
    # can resolve postponed type annotations correctly.
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    if not hasattr(module, "app"):
        raise RuntimeError(
            "The FastAPI service does not expose an 'app' object."
        )

    return module


_service_module = load_fastapi_service()
app = _service_module.app