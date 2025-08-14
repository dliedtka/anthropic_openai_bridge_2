import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import anthropic.types
import httpx
import pytest

from anthropic_openai_bridge import AnthropicOpenAIBridge
from anthropic_openai_bridge.config.config_manager import ConfigManager


class TestBridgeIntegration:
    @pytest.fixture
    def test_config(self):
        """Create a test configuration"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("OPENAI_API_KEY=test_openai_key\n")
            f.write("ANTHROPIC_API_KEY=test_anthropic_key\n")
            temp_env_file = f.name

        try:
            config = ConfigManager(env_file=temp_env_file)
            yield config
        finally:
            os.unlink(temp_env_file)

    @pytest.fixture
    def anthropic_samples(self):
        fixtures_path = (
            Path(__file__).parent.parent / "fixtures" / "anthropic_samples.json"
        )
        with open(fixtures_path) as f:
            return json.load(f)

    @pytest.fixture
    def openai_samples(self):
        fixtures_path = (
            Path(__file__).parent.parent / "fixtures" / "openai_samples.json"
        )
        with open(fixtures_path) as f:
            return json.load(f)

    def test_end_to_end_simple_conversation(
        self, test_config, anthropic_samples, openai_samples
    ):
        """Test complete end-to-end conversation flow"""
        bridge = AnthropicOpenAIBridge(config_manager=test_config)

        # Mock the OpenAI client to return our test response
        mock_response = openai_samples["simple_response"]

        with patch.object(
            bridge.openai_client, "create_chat_completion", return_value=mock_response
        ):
            # Send Anthropic request
            anthropic_request = anthropic_samples["simple_request"]
            result = bridge.send_message(anthropic_request)

            # Verify the response is an Anthropic Message object
            assert isinstance(result, anthropic.types.Message)
            assert result.role == "assistant"
            assert result.content[0].type == "text"
            assert result.usage is not None
            assert hasattr(result.usage, "input_tokens")
            assert hasattr(result.usage, "output_tokens")

    def test_end_to_end_with_system_message(
        self, test_config, anthropic_samples, openai_samples
    ):
        """Test conversation with system message"""
        bridge = AnthropicOpenAIBridge(config_manager=test_config)

        mock_response = openai_samples["simple_response"]

        with patch.object(
            bridge.openai_client, "create_chat_completion", return_value=mock_response
        ) as mock_client:
            # Send Anthropic request with system message
            anthropic_request = anthropic_samples["system_message_request"]
            result = bridge.send_message(anthropic_request)

            # Verify that the OpenAI client was called with converted request
            called_request = mock_client.call_args[0][0]
            assert len(called_request["messages"]) == 2
            assert called_request["messages"][0]["role"] == "system"
            assert called_request["messages"][1]["role"] == "user"

            # Verify response format
            assert isinstance(result, anthropic.types.Message)
            assert result.role == "assistant"

    def test_end_to_end_multi_turn(
        self, test_config, anthropic_samples, openai_samples
    ):
        """Test multi-turn conversation"""
        bridge = AnthropicOpenAIBridge(config_manager=test_config)

        mock_response = openai_samples["simple_response"]

        with patch.object(
            bridge.openai_client, "create_chat_completion", return_value=mock_response
        ) as mock_client:
            # Send multi-turn request
            anthropic_request = anthropic_samples["multi_turn_request"]
            result = bridge.send_message(anthropic_request)

            # Verify OpenAI client received multi-turn conversation
            called_request = mock_client.call_args[0][0]
            assert len(called_request["messages"]) == 3
            assert called_request["messages"][0]["role"] == "user"
            assert called_request["messages"][1]["role"] == "assistant"
            assert called_request["messages"][2]["role"] == "user"

            # Verify response
            assert isinstance(result, anthropic.types.Message)
            assert result.role == "assistant"

    def test_configuration_loading(self, test_config):
        """Test that bridge loads configuration properly"""
        bridge = AnthropicOpenAIBridge(config_manager=test_config)

        assert bridge.config.openai_api_key == "test_openai_key"
        assert bridge.config.anthropic_api_key == "test_anthropic_key"

    def test_error_handling(self, test_config):
        """Test that errors from OpenAI are properly handled"""
        bridge = AnthropicOpenAIBridge(config_manager=test_config)

        error_response = {
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        }

        with patch.object(
            bridge.openai_client, "create_chat_completion", return_value=error_response
        ):
            with pytest.raises(Exception, match="Invalid API key"):
                bridge.send_message(
                    {
                        "model": "claude-3-sonnet-20240229",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "test"}],
                    }
                )

    def test_bridge_with_custom_openai_api_key(self, anthropic_samples, openai_samples):
        """Test bridge with custom OpenAI API key"""
        bridge = AnthropicOpenAIBridge(openai_api_key="custom_test_key")

        assert bridge.config.openai_api_key == "custom_test_key"

        # Test that it works end-to-end
        mock_response = openai_samples["simple_response"]

        with patch.object(
            bridge.openai_client, "create_chat_completion", return_value=mock_response
        ):
            anthropic_request = anthropic_samples["simple_request"]
            result = bridge.send_message(anthropic_request)

            assert isinstance(result, anthropic.types.Message)
            assert result.role == "assistant"
            assert result.content is not None

    def test_bridge_with_custom_base_url(self, anthropic_samples, openai_samples):
        """Test bridge with custom OpenAI base URL"""
        bridge = AnthropicOpenAIBridge(
            openai_api_key="custom_test_key",
            openai_base_url="https://custom.endpoint.com",
        )

        assert bridge.config.openai_api_key == "custom_test_key"
        assert bridge.config.openai_base_url == "https://custom.endpoint.com"

        # Test that it works end-to-end
        mock_response = openai_samples["simple_response"]

        with patch.object(
            bridge.openai_client, "create_chat_completion", return_value=mock_response
        ):
            anthropic_request = anthropic_samples["simple_request"]
            result = bridge.send_message(anthropic_request)

            assert isinstance(result, anthropic.types.Message)
            assert result.role == "assistant"
            assert result.content is not None

    def test_bridge_with_custom_httpx_client(self, anthropic_samples, openai_samples):
        """Test bridge with custom httpx client"""
        custom_httpx_client = httpx.Client(timeout=30.0)

        bridge = AnthropicOpenAIBridge(
            openai_api_key="custom_test_key", httpx_client=custom_httpx_client
        )

        assert bridge.config.openai_api_key == "custom_test_key"
        assert bridge.config.httpx_client == custom_httpx_client

        # Test that it works end-to-end
        mock_response = openai_samples["simple_response"]

        with patch.object(
            bridge.openai_client, "create_chat_completion", return_value=mock_response
        ):
            anthropic_request = anthropic_samples["simple_request"]
            result = bridge.send_message(anthropic_request)

            assert isinstance(result, anthropic.types.Message)
            assert result.role == "assistant"
            assert result.content is not None

    def test_bridge_with_all_custom_parameters(self, anthropic_samples, openai_samples):
        """Test bridge with all custom parameters"""
        custom_httpx_client = httpx.Client(timeout=30.0)

        bridge = AnthropicOpenAIBridge(
            openai_api_key="custom_test_key",
            openai_base_url="https://custom.endpoint.com",
            httpx_client=custom_httpx_client,
        )

        assert bridge.config.openai_api_key == "custom_test_key"
        assert bridge.config.openai_base_url == "https://custom.endpoint.com"
        assert bridge.config.httpx_client == custom_httpx_client

        # Test that it works end-to-end
        mock_response = openai_samples["simple_response"]

        with patch.object(
            bridge.openai_client, "create_chat_completion", return_value=mock_response
        ):
            anthropic_request = anthropic_samples["simple_request"]
            result = bridge.send_message(anthropic_request)

            assert isinstance(result, anthropic.types.Message)
            assert result.role == "assistant"
            assert result.content is not None
