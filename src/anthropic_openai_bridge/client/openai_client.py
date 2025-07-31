from typing import Dict, Any
import openai
from ..config.config_manager import ConfigManager


class OpenAIClientWrapper:
    """Wrapper for OpenAI client with configuration management"""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize OpenAI client wrapper
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        
        # Initialize OpenAI client
        client_kwargs = {
            "api_key": self.config.openai_api_key
        }
        
        # Add custom base URL if specified
        if self.config.openai_base_url:
            client_kwargs["base_url"] = self.config.openai_base_url
        
        self.client = openai.OpenAI(**client_kwargs)
    
    def create_chat_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chat completion using OpenAI API
        
        Args:
            request: OpenAI-formatted request dictionary
            
        Returns:
            OpenAI response dictionary
        """
        try:
            response = self.client.chat.completions.create(**request)
            return response.model_dump()
        except Exception as e:
            # Convert OpenAI exceptions to consistent format
            return {
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": getattr(e, 'code', 'unknown')
                }
            }