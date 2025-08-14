"""Main bridge class for Anthropic-OpenAI API conversion"""

from typing import Any, Dict, Optional

import anthropic.types

from .client.openai_client import OpenAIClientWrapper
from .config.config_manager import ConfigManager
from .converters.request_converter import RequestConverter
from .converters.response_converter import ResponseConverter


class AnthropicOpenAIBridge:
    """Main bridge class that converts Anthropic requests to OpenAI and back"""

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        httpx_client: Optional[Any] = None,
    ):
        """Initialize the bridge with configuration and converters

        Args:
            config_manager: Optional configuration manager. If None, uses default.
            openai_api_key: Custom OpenAI API key (overrides environment variable)
            openai_base_url: Custom OpenAI base URL (overrides environment variable)
            httpx_client: Custom httpx client for OpenAI requests
        """
        # If custom parameters are provided but no config_manager, create one with the custom params
        if (
            openai_api_key or openai_base_url or httpx_client
        ) and config_manager is None:
            self.config = ConfigManager(
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                httpx_client=httpx_client,
            )
        else:
            self.config = config_manager or ConfigManager()

        self.openai_client = OpenAIClientWrapper(self.config)
        self.request_converter = RequestConverter()
        self.response_converter = ResponseConverter()

    def send_message(
        self, anthropic_request: Dict[str, Any]
    ) -> anthropic.types.Message:
        """Send a message through the bridge

        Args:
            anthropic_request: Anthropic Messages API request format

        Returns:
            Anthropic Messages API response as Message object
        """
        # Convert Anthropic request to OpenAI format
        openai_request = self.request_converter.convert(anthropic_request)

        # Send request to OpenAI API
        openai_response = self.openai_client.create_chat_completion(openai_request)

        # Convert OpenAI response back to Anthropic format
        anthropic_response = self.response_converter.convert(openai_response)

        return anthropic_response
