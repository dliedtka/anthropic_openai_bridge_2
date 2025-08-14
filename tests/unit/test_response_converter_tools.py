import json
from pathlib import Path

import anthropic.types
import pytest

from anthropic_openai_bridge.converters.response_converter import ResponseConverter


class TestResponseConverterTools:
    @pytest.fixture
    def converter(self):
        return ResponseConverter()

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

    def test_convert_response_with_single_tool_call(
        self, converter, openai_tool_samples
    ):
        """Test converting OpenAI response with single tool call"""
        openai_response = openai_tool_samples["tool_use_response"]

        result = converter.convert(openai_response)

        assert isinstance(result, anthropic.types.Message)
        assert result.role == "assistant"
        assert result.stop_reason == "tool_use"
        assert len(result.content) == 2  # text + tool_use

        # Check text content
        assert isinstance(result.content[0], anthropic.types.TextBlock)
        assert result.content[0].text == "I'll check the weather for you."

        # Check tool use content
        assert isinstance(result.content[1], anthropic.types.ToolUseBlock)
        assert result.content[1].name == "get_weather"
        assert result.content[1].id == "toolu_01D7FLrfh4GYq7yT1ULFeyMV"
        assert result.content[1].input == {
            "location": "San Francisco, CA",
            "unit": "fahrenheit",
        }

    def test_convert_response_with_multiple_tool_calls(
        self, converter, openai_tool_samples
    ):
        """Test converting OpenAI response with multiple tool calls"""
        openai_response = openai_tool_samples["multiple_tool_use_response"]

        result = converter.convert(openai_response)

        assert len(result.content) == 3  # text + 2 tool_use blocks
        assert isinstance(result.content[0], anthropic.types.TextBlock)
        assert isinstance(result.content[1], anthropic.types.ToolUseBlock)
        assert isinstance(result.content[2], anthropic.types.ToolUseBlock)

        # Check first tool call
        assert result.content[1].name == "get_weather"
        assert result.content[1].id == "toolu_01Weather123"
        assert result.content[1].input == {"location": "Boston, MA"}

        # Check second tool call
        assert result.content[2].name == "calculate"
        assert result.content[2].id == "toolu_01Calc456"
        assert result.content[2].input == {"expression": "15 * 8"}

    def test_convert_response_with_only_tool_calls_no_text(self, converter):
        """Test converting response with only tool calls, no text content"""
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,  # No text content
                        "tool_calls": [
                            {
                                "id": "call_12345",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"New York"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

        result = converter.convert(openai_response)

        assert len(result.content) == 1  # Only tool_use block
        assert isinstance(result.content[0], anthropic.types.ToolUseBlock)
        assert result.content[0].name == "get_weather"

    def test_convert_response_with_empty_content_and_tools(self, converter):
        """Test converting response with empty content but tool calls"""
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",  # Empty string content
                        "tool_calls": [
                            {
                                "id": "call_12345",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"Chicago"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

        result = converter.convert(openai_response)

        assert len(result.content) == 1  # Only tool_use (empty text content is skipped)
        assert isinstance(result.content[0], anthropic.types.ToolUseBlock)

    def test_convert_response_finish_reason_tool_calls(self, converter):
        """Test that finish_reason 'tool_calls' maps to stop_reason 'tool_use'"""
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Using tool",
                        "tool_calls": [
                            {
                                "id": "call_12345",
                                "type": "function",
                                "function": {"name": "test_tool", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        result = converter.convert(openai_response)

        assert result.stop_reason == "tool_use"

    def test_convert_response_no_tool_calls(self, converter):
        """Test converting regular response without tool calls still works"""
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

        result = converter.convert(openai_response)

        assert len(result.content) == 1
        assert isinstance(result.content[0], anthropic.types.TextBlock)
        assert result.content[0].text == "Hello! How can I help you?"
        assert result.stop_reason == "end_turn"

    def test_convert_response_invalid_tool_call_json(self, converter):
        """Test handling invalid JSON in tool call arguments"""
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Making call",
                        "tool_calls": [
                            {
                                "id": "call_12345",
                                "type": "function",
                                "function": {
                                    "name": "test_tool",
                                    "arguments": "{invalid json",  # Invalid JSON
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        result = converter.convert(openai_response)

        assert len(result.content) == 2
        assert isinstance(result.content[1], anthropic.types.ToolUseBlock)
        assert result.content[1].input == {}  # Should default to empty dict

    def test_convert_response_empty_message_creates_default_content(self, converter):
        """Test that completely empty message creates default empty text block"""
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        # No tool_calls either
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        }

        result = converter.convert(openai_response)

        assert len(result.content) == 1
        assert isinstance(result.content[0], anthropic.types.TextBlock)
        assert result.content[0].text == ""

    def test_convert_response_usage_mapping(self, converter, openai_tool_samples):
        """Test that usage tokens are correctly mapped"""
        openai_response = openai_tool_samples["tool_use_response"]

        result = converter.convert(openai_response)

        assert result.usage.input_tokens == 150
        assert result.usage.output_tokens == 75

    def test_convert_response_model_field(self, converter, openai_tool_samples):
        """Test that model field is correctly preserved"""
        openai_response = openai_tool_samples["tool_use_response"]

        result = converter.convert(openai_response)

        assert result.model == "claude-3-sonnet"
