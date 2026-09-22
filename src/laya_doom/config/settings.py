from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]

class AppSettings(BaseModel):
    scenario: Path = PROJECT_ROOT / "scenarios" / "v1_basic.cfg"
    model: str = "aac6fef/laya-mlx"
    dtype: str = "float16"
    episodes: int = Field(default=1, ge=1)
    logging_level: str = "INFO"
    telemetry_enabled: bool = True
    telemetry_root: Path = PROJECT_ROOT / "runs"
