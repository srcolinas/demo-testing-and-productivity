from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    models_dir: Path
