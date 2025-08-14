import json
from pathlib import Path

import anthropic.types
import pytest

from anthropic_openai_bridge.converters.tool_converter import ToolConverter


class TestToolConverter:
    @pytest.fixture
    def converter(self):
        return ToolConverter()

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

    def test_convert_anthropic_tools_to_openai_simple(
        self, converter, anthropic_tool_samples, openai_tool_samples
    ):
        """Test converting simple tool definition from Anthropic to OpenAI format"""
        anthropic_tools = anthropic_tool_samples["simple_tool_request"]["tools"]
        expected_openai_tools = openai_tool_samples["simple_tool_request"]["tools"]

        result = converter.convert_anthropic_tools_to_openai(anthropic_tools)

        assert result == expected_openai_tools

    def test_convert_anthropic_tools_to_openai_multiple(
        self, converter, anthropic_tool_samples, openai_tool_samples
    ):
        """Test converting multiple tool definitions"""
        anthropic_tools = anthropic_tool_samples["multiple_tools_request"]["tools"]
        expected_openai_tools = openai_tool_samples["multiple_tools_request"]["tools"]

        result = converter.convert_anthropic_tools_to_openai(anthropic_tools)

        assert result == expected_openai_tools

    def test_convert_openai_tool_calls_to_anthropic_single(self, converter):
        """Test converting single OpenAI tool call to Anthropic tool use block"""
        openai_tool_calls = [
            {
                "id": "call_12345",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location":"San Francisco, CA","unit":"fahrenheit"}',
                },
            }
        ]

        result = converter.convert_openai_tool_calls_to_anthropic(openai_tool_calls)

        assert len(result) == 1
        assert isinstance(result[0], anthropic.types.ToolUseBlock)
        assert result[0].type == "tool_use"
        assert result[0].id == "toolu_12345"
        assert result[0].name == "get_weather"
        assert result[0].input == {
            "location": "San Francisco, CA",
            "unit": "fahrenheit",
        }

    def test_convert_openai_tool_calls_to_anthropic_multiple(self, converter):
        """Test converting multiple OpenAI tool calls"""
        openai_tool_calls = [
            {
                "id": "call_weather123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location":"Boston, MA"}',
                },
            },
            {
                "id": "call_calc456",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": '{"expression":"15 * 8"}',
                },
            },
        ]

        result = converter.convert_openai_tool_calls_to_anthropic(openai_tool_calls)

        assert len(result) == 2
        assert result[0].name == "get_weather"
        assert result[0].input == {"location": "Boston, MA"}
        assert result[1].name == "calculate"
        assert result[1].input == {"expression": "15 * 8"}

    def test_convert_openai_tool_calls_invalid_json(self, converter):
        """Test handling invalid JSON in OpenAI function arguments"""
        openai_tool_calls = [
            {
                "id": "call_12345",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{invalid json"},
            }
        ]

        result = converter.convert_openai_tool_calls_to_anthropic(openai_tool_calls)

        assert len(result) == 1
        assert result[0].input == {}  # Should default to empty dict

    def test_convert_anthropic_tool_results_to_openai_single(self, converter):
        """Test converting single tool result"""
        anthropic_content = [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_12345",
                "content": "68°F, partly cloudy",
            }
        ]

        result = converter.convert_anthropic_tool_results_to_openai(anthropic_content)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_12345"
        assert result[0]["content"] == "68°F, partly cloudy"

    def test_convert_anthropic_tool_results_to_openai_multiple(self, converter):
        """Test converting multiple tool results"""
        anthropic_content = [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_weather123",
                "content": "42°F, snow showers",
            },
            {"type": "tool_result", "tool_use_id": "toolu_calc456", "content": "120"},
        ]

        result = converter.convert_anthropic_tool_results_to_openai(anthropic_content)

        assert len(result) == 2
        assert result[0]["tool_call_id"] == "call_weather123"
        assert result[0]["content"] == "42°F, snow showers"
        assert result[1]["tool_call_id"] == "call_calc456"
        assert result[1]["content"] == "120"

    def test_convert_tool_choice_auto(self, converter):
        """Test converting tool_choice 'auto'"""
        result = converter.convert_tool_choice_to_openai("auto")
        assert result == "auto"

    def test_convert_tool_choice_any(self, converter):
        """Test converting tool_choice 'any' to 'required'"""
        result = converter.convert_tool_choice_to_openai("any")
        assert result == "required"

    def test_convert_tool_choice_specific_tool(self, converter):
        """Test converting specific tool choice"""
        anthropic_choice = {"type": "tool", "name": "get_weather"}

        result = converter.convert_tool_choice_to_openai(anthropic_choice)

        expected = {"type": "function", "function": {"name": "get_weather"}}
        assert result == expected

    def test_tool_id_conversion_roundtrip(self, converter):
        """Test that tool ID conversion works in both directions"""
        openai_id = "call_12345abc"
        anthropic_id = converter._convert_tool_call_id(openai_id)
        back_to_openai = converter._convert_tool_use_id_to_call_id(anthropic_id)

        assert anthropic_id == "toolu_12345abc"
        assert back_to_openai == openai_id

    def test_generate_tool_call_id(self, converter):
        """Test tool call ID generation"""
        tool_id = converter.generate_tool_call_id()
        assert tool_id.startswith("call_")
        assert len(tool_id) == 13  # "call_" + 8 hex chars

    def test_generate_tool_use_id(self, converter):
        """Test tool use ID generation"""
        tool_id = converter.generate_tool_use_id()
        assert tool_id.startswith("toolu_")
        assert len(tool_id) == 18  # "toolu_" + 12 hex chars

    def test_convert_anthropic_tool_results_filters_non_tool_results(self, converter):
        """Test that only tool_result blocks are converted"""
        anthropic_content = [
            {"type": "text", "text": "Some text"},
            {"type": "tool_result", "tool_use_id": "toolu_12345", "content": "Result"},
        ]

        result = converter.convert_anthropic_tool_results_to_openai(anthropic_content)

        assert len(result) == 1  # Only tool_result should be converted
        assert result[0]["role"] == "tool"
