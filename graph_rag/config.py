from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    eval_judge_model: str = Field(default="gpt-4.1-mini", alias="EVAL_JUDGE_MODEL")
    token_encoding: str = Field(default="cl100k_base", alias="TOKEN_ENCODING")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    fuzzy_match_threshold: float = Field(default=90.0, alias="FUZZY_MATCH_THRESHOLD")
    local_search_top_k: int = Field(default=3, alias="LOCAL_SEARCH_TOP_K")
    local_search_max_hops: int = Field(default=1, alias="LOCAL_SEARCH_MAX_HOPS")
    local_search_max_chunks: int = Field(default=4, alias="LOCAL_SEARCH_MAX_CHUNKS")
    global_search_top_k: int = Field(default=2, alias="GLOBAL_SEARCH_TOP_K")
    enable_neo4j_sync: bool = Field(default=False, alias="ENABLE_NEO4J_SYNC")
    neo4j_uri: str | None = Field(default=None, alias="NEO4J_URI")
    neo4j_username: str | None = Field(default=None, alias="NEO4J_USERNAME")
    neo4j_password: SecretStr | None = Field(default=None, alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()