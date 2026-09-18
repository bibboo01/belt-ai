from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Belt AI"
    app_version: str = "1.0.0"
    environment: str = "development"

    database_url: str

    model_path: str = "models/production/best.pt"
    model_confidence: float = 0.5

    image_dir: str = "images"
    result_dir: str = "images/result"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def model_full_path(self) -> Path:
        return BASE_DIR / self.model_path


settings = Settings()