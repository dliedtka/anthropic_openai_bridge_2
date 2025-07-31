import json
import pytest
from pathlib import Path

from anthropic_openai_bridge.converters.response_converter import ResponseConverter


class TestResponseConverter:
    @pytest.fixture
    def converter(self):
        return ResponseConverter()

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

    def test_convert_simple_response(self, converter, openai_samples, anthropic_samples):
        openai_response = openai_samples["simple_response"]
        expected_anthropic_response = anthropic_samples["simple_response"]
        
        result = converter.convert(openai_response)
        
        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == expected_anthropic_response["content"][0]["text"]
        assert result["stop_reason"] == "end_turn"

    def test_convert_usage_stats(self, converter, openai_samples):
        openai_response = openai_samples["simple_response"]
        
        result = converter.convert(openai_response)
        
        assert "usage" in result
        assert result["usage"]["input_tokens"] == openai_response["usage"]["prompt_tokens"]
        assert result["usage"]["output_tokens"] == openai_response["usage"]["completion_tokens"]

    def test_convert_stop_reason(self, converter):
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
                        "content": "Test response"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        result = converter.convert(openai_response)
        
        assert result["stop_reason"] == "end_turn"

    def test_convert_length_stop_reason(self, converter):
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
                        "content": "Test response"
                    },
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        result = converter.convert(openai_response)
        
        assert result["stop_reason"] == "max_tokens"

    def test_handle_error_response(self, converter):
        error_response = {
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        }
        
        with pytest.raises(Exception) as exc_info:
            converter.convert(error_response)
        
        assert "Invalid API key" in str(exc_info.value)