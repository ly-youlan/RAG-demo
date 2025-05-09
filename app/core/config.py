from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str
    
    # OpenAI API 配置
    # OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENAI_API_BASE: str = "https://openrouter.ai/api/v1"
    OPENAI_API_KEY: str = ""

    # AI 服务配置
    AI_MODEL: str = "gpt-4o-mini"
    AI_TEMPERATURE: float = 0.7
    VISION_MODEL: str = "gpt-4o"
    LLAMA_INDEX_MODEL: str = "gpt-4o-mini"
    
    # DeepSeek API 配置
    DEEPSEEK_API_BASE: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-reasoner"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_TEMPERATURE: float = 0.7
    
    # MongoDB 配置
    MONGODB_URL: str = "mongodb+srv://fastapi:8l2YvJ6zHSSt7T24@cluster0.75lx5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    MONGODB_DB_NAME: str = "UIdata"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 