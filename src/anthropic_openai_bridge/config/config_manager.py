import os
from typing import Optional, Any
from dotenv import load_dotenv


class ConfigManager:
    """Manages configuration including API keys and endpoints"""
    
    def __init__(
        self, 
        env_file: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        httpx_client: Optional[Any] = None
    ):
        """Initialize configuration manager
        
        Args:
            env_file: Path to .env file. If None, uses default .env in project root.
            openai_api_key: Custom OpenAI API key (overrides environment variable)
            openai_base_url: Custom OpenAI base URL (overrides environment variable)
            httpx_client: Custom httpx client for OpenAI requests
        """
        # Load environment variables first
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Store custom parameters (these override environment variables)
        self._custom_openai_api_key = openai_api_key
        self._custom_openai_base_url = openai_base_url
        self._httpx_client = httpx_client
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from custom parameter or environment"""
        if self._custom_openai_api_key:
            return self._custom_openai_api_key
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required or must be provided as parameter")
        return api_key
    
    @property
    def openai_base_url(self) -> Optional[str]:
        """Get OpenAI base URL from custom parameter or environment"""
        if self._custom_openai_base_url:
            return self._custom_openai_base_url
        return os.getenv("OPENAI_BASE_URL")
    
    @property
    def httpx_client(self) -> Optional[Any]:
        """Get custom httpx client if provided"""
        return self._httpx_client
    
    @property
    def anthropic_api_key(self) -> str:
        """Get Anthropic API key from environment (for reference/testing)"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        return api_key