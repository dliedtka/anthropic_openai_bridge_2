import json
from pathlib import Path

import pytest

from anthropic_openai_bridge.converters.request_converter import RequestConverter


class TestRequestConverterTools:
    @pytest.fixture
    def converter(self):
        return RequestConverter()

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

    def test_convert_request_with_simple_tool(
        self, converter, anthropic_tool_samples, openai_tool_samples
    ):
        """Test converting request with a single tool definition"""
        anthropic_request = anthropic_tool_samples["simple_tool_request"]
        expected_openai_request = openai_tool_samples["simple_tool_request"]

        result = converter.convert(anthropic_request)

        assert result["model"] == expected_openai_request["model"]
        assert (
            result["max_completion_tokens"]
            == expected_openai_request["max_completion_tokens"]
        )
        assert result["tools"] == expected_openai_request["tools"]
        assert result["messages"] == expected_openai_request["messages"]

    def test_convert_request_with_multiple_tools(
        self, converter, anthropic_tool_samples, openai_tool_samples
    ):
        """Test converting request with multiple tool definitions"""
        anthropic_request = anthropic_tool_samples["multiple_tools_request"]
        expected_openai_request = openai_tool_samples["multiple_tools_request"]

        result = converter.convert(anthropic_request)

        assert result["tools"] == expected_openai_request["tools"]
        assert len(result["tools"]) == 2

    def test_convert_request_with_tool_choice_specific(
        self, converter, anthropic_tool_samples, openai_tool_samples
    ):
        """Test converting request with specific tool choice"""
        anthropic_request = anthropic_tool_samples["tool_choice_specific"]
        expected_openai_request = openai_tool_samples["tool_choice_specific"]

        result = converter.convert(anthropic_request)

        assert result["tool_choice"] == expected_openai_request["tool_choice"]

    def test_convert_request_with_tool_choice_auto(self, converter):
        """Test converting request with tool_choice 'auto'"""
        anthropic_request = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "tool_choice": "auto",
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = converter.convert(anthropic_request)

        assert result["tool_choice"] == "auto"

    def test_convert_messages_with_tool_results_single(
        self, converter, anthropic_tool_samples
    ):
        """Test converting messages containing tool results"""
        anthropic_request = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                anthropic_tool_samples["tool_result_message"],
            ],
        }

        result = converter.convert(anthropic_request)

        # Should have user message + tool message
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "What's the weather?"
        assert result["messages"][1]["role"] == "tool"
        assert result["messages"][1]["tool_call_id"] == "call_01D7FLrfh4GYq7yT1ULFeyMV"

    def test_convert_messages_with_multiple_tool_results(
        self, converter, anthropic_tool_samples
    ):
        """Test converting messages with multiple tool results"""
        anthropic_request = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Get weather and calculate"},
                anthropic_tool_samples["multiple_tool_results_message"],
            ],
        }

        result = converter.convert(anthropic_request)

        # Should have user message + 2 tool messages
        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "tool"
        assert result["messages"][1]["tool_call_id"] == "call_01Weather123"
        assert result["messages"][2]["role"] == "tool"
        assert result["messages"][2]["tool_call_id"] == "call_01Calc456"

    def test_convert_messages_with_text_and_tool_result(
        self, converter, anthropic_tool_samples
    ):
        """Test converting messages with both text and tool results"""
        anthropic_request = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "messages": [anthropic_tool_samples["text_and_tool_result"]],
        }

        result = converter.convert(anthropic_request)

        # Should have both user text message and tool message
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert (
            result["messages"][0]["content"]
            == "Thanks for checking! Here's the result:"
        )
        assert result["messages"][1]["role"] == "tool"

    def test_convert_multi_turn_tool_conversation(
        self, converter, anthropic_tool_samples, openai_tool_samples
    ):
        """Test converting full multi-turn tool conversation"""
        anthropic_request = anthropic_tool_samples["multi_turn_tool_conversation"]
        expected_openai_request = openai_tool_samples["multi_turn_tool_conversation"]

        result = converter.convert(anthropic_request)

        assert result["tools"] == expected_openai_request["tools"]
        assert result["messages"] == expected_openai_request["messages"]

    def test_convert_request_without_tools(self, converter):
        """Test that requests without tools still work"""
        anthropic_request = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        result = converter.convert(anthropic_request)

        assert "tools" not in result
        assert "tool_choice" not in result
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello!"

    def test_convert_message_with_empty_content_list(self, converter):
        """Test handling messages with empty content list"""
        anthropic_request = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": []}],
        }

        result = converter.convert(anthropic_request)

        # Should handle gracefully - empty list should not create messages
        assert len(result["messages"]) == 0

    def test_convert_message_with_text_blocks(self, converter):
        """Test converting messages with text blocks"""
        anthropic_request = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ],
                }
            ],
        }

        result = converter.convert(anthropic_request)

        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello World"
