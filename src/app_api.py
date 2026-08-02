from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SERVICE_PATH = ROOT_DIR / "19_fastapi_service.py"

spec = importlib.util.spec_from_file_location(
    "hevemind_fastapi_service",
    SERVICE_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Unable to load HeveMind API service from {SERVICE_PATH}"
    )

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

app = module.app