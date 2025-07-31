import json
import pytest
from pathlib import Path

from anthropic_openai_bridge.converters.request_converter import RequestConverter


class TestRequestConverter:
    @pytest.fixture
    def converter(self):
        return RequestConverter()

    @pytest.fixture
    def anthropic_samples(self):
        fixtures_path = Path(__file__).parent.parent / "fixtures" / "anthropic_samples.json"
        with open(fixtures_path) as f:
            return json.load(f)

    @pytest.fixture
    def openai_samples(self):
        fixtures_path = Path(__file__).parent.parent / "fixtures" / "openai_samples.json"
        with open(fixtures_path) as f:
            return json.load(f)

    def test_convert_simple_user_message(self, converter, anthropic_samples, openai_samples):
        anthropic_request = anthropic_samples["simple_request"]
        expected_openai_request = openai_samples["simple_request"]
        
        result = converter.convert(anthropic_request)
        
        assert result["messages"] == expected_openai_request["messages"]
        assert result["model"] == expected_openai_request["model"]
        assert result["max_completion_tokens"] == expected_openai_request["max_completion_tokens"]

    def test_convert_system_message(self, converter, anthropic_samples, openai_samples):
        anthropic_request = anthropic_samples["system_message_request"]
        expected_openai_request = openai_samples["system_message_request"]
        
        result = converter.convert(anthropic_request)
        
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are a helpful assistant."
        assert result["messages"][1]["role"] == "user"
        assert result["messages"] == expected_openai_request["messages"]

    def test_convert_multi_turn_conversation(self, converter, anthropic_samples, openai_samples):
        anthropic_request = anthropic_samples["multi_turn_request"]
        expected_openai_request = openai_samples["multi_turn_request"]
        
        result = converter.convert(anthropic_request)
        
        assert len(result["messages"]) == 3
        assert result["messages"] == expected_openai_request["messages"]

    def test_convert_model_parameter_passthrough(self, converter):
        # Test with custom model name to ensure no mapping occurs
        anthropic_request = {
            "model": "our_custom_model_123",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "test"}]
        }
        
        result = converter.convert(anthropic_request)
        
        assert result["model"] == "our_custom_model_123"
        
        # Test with OpenAI-style model name to ensure it passes through
        anthropic_request_openai_style = {
            "model": "gpt-4-turbo",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "test"}]
        }
        
        result = converter.convert(anthropic_request_openai_style)
        
        assert result["model"] == "gpt-4-turbo"

    def test_convert_missing_model_parameter(self, converter):
        # Test that missing model parameter is handled gracefully
        anthropic_request = {
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "test"}]
        }
        
        result = converter.convert(anthropic_request)
        
        assert "model" not in result

    def test_convert_max_tokens_parameter(self, converter):
        anthropic_request = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "test"}]
        }
        
        result = converter.convert(anthropic_request)
        
        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 2048

    def test_convert_temperature_parameter(self, converter):
        anthropic_request = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "test"}]
        }
        
        result = converter.convert(anthropic_request)
        
        assert result["temperature"] == 0.7