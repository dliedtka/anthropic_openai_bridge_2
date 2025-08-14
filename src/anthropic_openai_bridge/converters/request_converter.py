from typing import Any, Dict, List


class RequestConverter:
    """Converts Anthropic Messages API requests to OpenAI ChatCompletions API requests"""

    def convert(self, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an Anthropic request to OpenAI format"""
        openai_request = {}

        # Pass through model name unchanged
        if "model" in anthropic_request:
            openai_request["model"] = anthropic_request["model"]

        # Convert max_tokens to max_completion_tokens
        if "max_tokens" in anthropic_request:
            openai_request["max_completion_tokens"] = anthropic_request["max_tokens"]

        # Pass through temperature if present
        if "temperature" in anthropic_request:
            openai_request["temperature"] = anthropic_request["temperature"]

        # Convert messages
        messages = []

        # Handle system message
        if "system" in anthropic_request:
            messages.append({"role": "system", "content": anthropic_request["system"]})

        # Add user/assistant messages
        for message in anthropic_request.get("messages", []):
            messages.append({"role": message["role"], "content": message["content"]})

        openai_request["messages"] = messages

        return openai_request
