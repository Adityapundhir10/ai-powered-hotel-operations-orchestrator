from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name: str = "AI-Powered Hotel Operations Workflow Orchestrator"
    app_env: str = "development"
    author_name: str = "Aditya Pundhir"
    database_url: str = ""
    redis_url: str = ""
    kafka_bootstrap_servers: str = ""
    kafka_topic: str = "hotel-operations-events"
    weaviate_http_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051
    use_weaviate: bool = False
    use_transformers: bool = False
    use_langgraph: bool = True
    distilbert_model_path: str = "models/distilbert-checkpoint"
    minilm_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    layoutlm_model_name: str = "microsoft/layoutlmv3-base"
    layoutlm_model_path: str = "models/layoutlmv3-checkpoint"
    forecast_model_path: str = "models/occupancy_xgb.joblib"
    sop_directory: str = "data/sops"
    default_sla_minutes: int = 30
    human_approval_severity: str = "critical"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else self.project_root / path


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
