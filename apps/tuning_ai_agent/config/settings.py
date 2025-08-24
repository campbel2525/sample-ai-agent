from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 設定
    debugpy_port: int = 9000  # デバッグ用ポート
    # test_data_path: str = "agent/data/test_data.yml"  # テストデータのパス
    ai_agent_api_url: str = ""  # AIエージェントのAPI URL
    # tuning_result_dir: str = "data/tuning_result/"  # チューニング結果のディレクトリ

    # OpenAI
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-2024-08-06"
