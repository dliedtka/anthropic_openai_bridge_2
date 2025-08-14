import json
from pathlib import Path
from unittest.mock import Mock, patch

import anthropic.types
import pytest

from anthropic_openai_bridge import AnthropicOpenAIBridge


class TestToolCallingIntegration:
    @pytest.fixture
    def anthropic_tool_samples(self):
        fixtures_path = (
            Path(__file__).parent.parent
            / "fixtures"
            / "tool_calling_anthropic_samples.json"
        )
        with open(fixtures_path) as f:
            return json.load(f)

    @pytest.fixture
    def openai_tool_samples(self):
        fixtures_path = (
            Path(__file__).parent.parent
            / "fixtures"
            / "tool_calling_openai_samples.json"
        )
        with open(fixtures_path) as f:
            return json.load(f)

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.openai_api_key = "test-api-key"
        config.openai_base_url = "https://api.openai.com/v1"
        config.httpx_client = None
        return config

    @pytest.fixture
    def bridge(self, mock_config):
        with patch(
            "anthropic_openai_bridge.config.config_manager.ConfigManager"
        ) as mock_cm:
            mock_cm.return_value = mock_config
            bridge = AnthropicOpenAIBridge()
            return bridge

    def test_simple_tool_calling_end_to_end(
        self, bridge, anthropic_tool_samples, openai_tool_samples
    ):
        """Test complete tool calling workflow with single tool"""
        anthropic_request = anthropic_tool_samples["simple_tool_request"]
        openai_response = openai_tool_samples["tool_use_response"]

        # Mock the OpenAI client response
        with patch.object(
            bridge.openai_client, "create_chat_completion"
        ) as mock_create:
            mock_create.return_value = openai_response

            result = bridge.send_message(anthropic_request)

            # Verify the request was properly converted and sent
            mock_create.assert_called_once()
            sent_request = mock_create.call_args[0][0]

            # Check that tools were converted to OpenAI format
            assert "tools" in sent_request
            assert (
                sent_request["tools"]
                == openai_tool_samples["simple_tool_request"]["tools"]
            )

            # Verify the response is proper Anthropic format
            assert isinstance(result, anthropic.types.Message)
            assert result.stop_reason == "tool_use"
            assert len(result.content) == 2  # text + tool_use

            tool_use = result.content[1]
            assert isinstance(tool_use, anthropic.types.ToolUseBlock)
            assert tool_use.name == "get_weather"
            assert tool_use.input == {
                "location": "San Francisco, CA",
                "unit": "fahrenheit",
            }

    def test_multiple_tool_calling_end_to_end(
        self, bridge, anthropic_tool_samples, openai_tool_samples
    ):
        """Test tool calling with multiple tools defined"""
        anthropic_request = anthropic_tool_samples["multiple_tools_request"]
        openai_response = openai_tool_samples["multiple_tool_use_response"]

        with patch.object(
            bridge.openai_client, "create_chat_completion"
        ) as mock_create:
            mock_create.return_value = openai_response

            result = bridge.send_message(anthropic_request)

            # Check that multiple tools were converted
            sent_request = mock_create.call_args[0][0]
            assert len(sent_request["tools"]) == 2

            # Verify multiple tool use blocks in response
            assert len(result.content) == 3  # text + 2 tool_use blocks
            assert isinstance(result.content[1], anthropic.types.ToolUseBlock)
            assert isinstance(result.content[2], anthropic.types.ToolUseBlock)
            assert result.content[1].name == "get_weather"
            assert result.content[2].name == "calculate"

    def test_tool_choice_specific_end_to_end(
        self, bridge, anthropic_tool_samples, openai_tool_samples
    ):
        """Test tool calling with specific tool choice"""
        anthropic_request = anthropic_tool_samples["tool_choice_specific"]
        openai_response = openai_tool_samples["tool_use_response"]

        with patch.object(
            bridge.openai_client, "create_chat_completion"
        ) as mock_create:
            mock_create.return_value = openai_response

            result = bridge.send_message(anthropic_request)

            # Check that tool_choice was converted
            sent_request = mock_create.call_args[0][0]
            assert "tool_choice" in sent_request
            expected_tool_choice = openai_tool_samples["tool_choice_specific"][
                "tool_choice"
            ]
            assert sent_request["tool_choice"] == expected_tool_choice

    def test_multi_turn_tool_conversation_end_to_end(
        self, bridge, anthropic_tool_samples, openai_tool_samples
    ):
        """Test complete multi-turn conversation with tool calls and results"""
        anthropic_request = anthropic_tool_samples["multi_turn_tool_conversation"]
        openai_response = {
            "id": "chatcmpl-final",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "claude-3-sonnet",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Based on the weather data, it's 68°F and partly cloudy in San Francisco.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 25,
                "total_tokens": 225,
            },
        }

        with patch.object(
            bridge.openai_client, "create_chat_completion"
        ) as mock_create:
            mock_create.return_value = openai_response

            result = bridge.send_message(anthropic_request)

            # Check that the conversation was properly converted
            sent_request = mock_create.call_args[0][0]
            expected_messages = openai_tool_samples["multi_turn_tool_conversation"][
                "messages"
            ]
            assert sent_request["messages"] == expected_messages

            # Verify final response
            assert isinstance(result, anthropic.types.Message)
            assert result.stop_reason == "end_turn"
            assert len(result.content) == 1
            assert "68°F and partly cloudy" in result.content[0].text

    def test_tool_result_conversion_in_messages(self, bridge, anthropic_tool_samples):
        """Test that tool results in messages are properly converted"""
        # Create request with tool result in messages
        anthropic_request = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll check for you."},
                        {
                            "type": "tool_use",
                            "id": "toolu_12345",
                            "name": "get_weather",
                            "input": {"location": "San Francisco"},
                        },
                    ],
                },
                anthropic_tool_samples["tool_result_message"],
            ],
        }

        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "claude-3-sonnet",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "The weather is nice!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 10,
                "total_tokens": 160,
            },
        }

        with patch.object(
            bridge.openai_client, "create_chat_completion"
        ) as mock_create:
            mock_create.return_value = openai_response

            result = bridge.send_message(anthropic_request)

            # Check that tool results were converted to tool role messages
            sent_request = mock_create.call_args[0][0]
            messages = sent_request["messages"]

            # Should have: user, assistant, tool messages
            tool_messages = [msg for msg in messages if msg["role"] == "tool"]
            assert len(tool_messages) == 1
            assert tool_messages[0]["tool_call_id"] == "call_01D7FLrfh4GYq7yT1ULFeyMV"

    def test_error_handling_in_tool_context(self, bridge):
        """Test error handling during tool calling operations"""
        anthropic_request = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "tools": [
                {
                    "name": "test_tool",
                    "description": "Test tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {"param": {"type": "string"}},
                        "required": ["param"],
                    },
                }
            ],
            "messages": [{"role": "user", "content": "Use the tool"}],
        }

        # Mock an error response
        error_response = {
            "error": {"message": "Invalid tool call", "type": "invalid_request_error"}
        }

        with patch.object(
            bridge.openai_client, "create_chat_completion"
        ) as mock_create:
            mock_create.return_value = error_response

            with pytest.raises(Exception) as exc_info:
                bridge.send_message(anthropic_request)

            assert "OpenAI API Error: Invalid tool call" in str(exc_info.value)

    def test_backward_compatibility_non_tool_requests(self, bridge):
        """Test that non-tool requests still work as before"""
        anthropic_request = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
        }

        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "claude-3-sonnet",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'm doing well, thank you!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }

        with patch.object(
            bridge.openai_client, "create_chat_completion"
        ) as mock_create:
            mock_create.return_value = openai_response

            result = bridge.send_message(anthropic_request)

            # Should work exactly like before
            assert isinstance(result, anthropic.types.Message)
            assert len(result.content) == 1
            assert isinstance(result.content[0], anthropic.types.TextBlock)
            assert result.content[0].text == "I'm doing well, thank you!"
            assert result.stop_reason == "end_turn"

            # No tools should be in the sent request
            sent_request = mock_create.call_args[0][0]
            assert "tools" not in sent_request
            assert "tool_choice" not in sent_request

    def test_empty_tool_list_handling(self, bridge):
        """Test handling of empty tool list"""
        anthropic_request = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "tools": [],  # Empty tools list
            "messages": [{"role": "user", "content": "Hello"}],
        }

        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "claude-3-sonnet",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }

        with patch.object(
            bridge.openai_client, "create_chat_completion"
        ) as mock_create:
            mock_create.return_value = openai_response

            result = bridge.send_message(anthropic_request)

            # Empty tools list should still be sent
            sent_request = mock_create.call_args[0][0]
            assert sent_request["tools"] == []
