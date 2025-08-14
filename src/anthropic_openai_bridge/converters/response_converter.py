import uuid
from typing import Any, Dict, List, cast

import anthropic.types


class ResponseConverter:
    """Converts OpenAI ChatCompletions API responses to Anthropic Messages API responses"""

    STOP_REASON_MAPPING = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
        "function_call": "tool_use",
        "tool_calls": "tool_use",
    }

    def convert(self, openai_response: Dict[str, Any]) -> anthropic.types.Message:
        """Convert an OpenAI response to Anthropic format"""

        # Handle error responses
        if "error" in openai_response:
            error_msg = openai_response["error"].get("message", "Unknown error")
            raise Exception(f"OpenAI API Error: {error_msg}")

        choice = openai_response["choices"][0]
        message = choice["message"]

        # Build Anthropic Message object
        content = [anthropic.types.TextBlock(type="text", text=message["content"])]

        usage = anthropic.types.Usage(
            input_tokens=openai_response.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=openai_response.get("usage", {}).get("completion_tokens", 0),
        )

        anthropic_response = anthropic.types.Message(
            id=f"msg_{uuid.uuid4().hex[:8].upper()}",
            type="message",
            role="assistant",
            content=cast(List[anthropic.types.ContentBlock], content),
            model=openai_response.get("model", "unknown"),
            stop_reason=cast(
                anthropic.types.StopReason,
                self._convert_stop_reason(choice.get("finish_reason", "stop")),
            ),
            stop_sequence=None,
            usage=usage,
        )

        return anthropic_response

    def _convert_stop_reason(self, openai_finish_reason: str) -> str:
        """Convert OpenAI finish_reason to Anthropic stop_reason"""
        return self.STOP_REASON_MAPPING.get(openai_finish_reason, "end_turn")

    def _convert_usage(self, openai_usage: Dict[str, Any]) -> anthropic.types.Usage:
        """Convert OpenAI usage stats to Anthropic format"""
        return anthropic.types.Usage(
            input_tokens=openai_usage.get("prompt_tokens", 0),
            output_tokens=openai_usage.get("completion_tokens", 0),
        )
