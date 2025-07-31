import os
from typing import Optional
from dotenv import load_dotenv


class ConfigManager:
    """Manages configuration including API keys and endpoints"""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration manager
        
        Args:
            env_file: Path to .env file. If None, uses default .env in project root.
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from environment"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return api_key
    
    @property
    def openai_base_url(self) -> Optional[str]:
        """Get OpenAI base URL from environment (for custom endpoints)"""
        return os.getenv("OPENAI_BASE_URL")
    
    @property
    def anthropic_api_key(self) -> str:
        """Get Anthropic API key from environment (for reference/testing)"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        return api_key