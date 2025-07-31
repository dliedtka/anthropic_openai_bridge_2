"""Main bridge class for Anthropic-OpenAI API conversion"""

from typing import Dict, Any, Optional
from .config.config_manager import ConfigManager
from .client.openai_client import OpenAIClientWrapper
from .converters.request_converter import RequestConverter
from .converters.response_converter import ResponseConverter


class AnthropicOpenAIBridge:
    """Main bridge class that converts Anthropic requests to OpenAI and back"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize the bridge with configuration and converters
        
        Args:
            config_manager: Optional configuration manager. If None, uses default.
        """
        self.config = config_manager or ConfigManager()
        self.openai_client = OpenAIClientWrapper(self.config)
        self.request_converter = RequestConverter()
        self.response_converter = ResponseConverter()
    
    def send_message(self, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message through the bridge
        
        Args:
            anthropic_request: Anthropic Messages API request format
            
        Returns:
            Anthropic Messages API response format
        """
        # Convert Anthropic request to OpenAI format
        openai_request = self.request_converter.convert(anthropic_request)
        
        # Send request to OpenAI API
        openai_response = self.openai_client.create_chat_completion(openai_request)
        
        # Convert OpenAI response back to Anthropic format
        anthropic_response = self.response_converter.convert(openai_response)
        
        return anthropic_response