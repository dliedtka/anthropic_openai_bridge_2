import uuid
from typing import Any, Dict, List, cast

import anthropic.types

from .tool_converter import ToolConverter


class ResponseConverter:
    """Converts OpenAI ChatCompletions API responses to Anthropic Messages API responses"""

    def __init__(self) -> None:
        self.tool_converter = ToolConverter()

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

        # Build content blocks
        content = self._build_content_blocks(message)

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

    def _build_content_blocks(
        self, message: Dict[str, Any]
    ) -> List[anthropic.types.ContentBlock]:
        """Build content blocks from OpenAI message, handling both text and tool calls"""
        content: List[anthropic.types.ContentBlock] = []

        # Add text content if present
        if message.get("content"):
            content.append(
                anthropic.types.TextBlock(type="text", text=message["content"])
            )

        # Add tool use blocks if present
        if "tool_calls" in message and message["tool_calls"]:
            tool_use_blocks = (
                self.tool_converter.convert_openai_tool_calls_to_anthropic(
                    message["tool_calls"]
                )
            )
            content.extend(tool_use_blocks)

        # If no content was created, add a default empty text block
        if not content:
            content.append(anthropic.types.TextBlock(type="text", text=""))

        return content

    def _convert_stop_reason(self, openai_finish_reason: str) -> str:
        """Convert OpenAI finish_reason to Anthropic stop_reason"""
        return self.STOP_REASON_MAPPING.get(openai_finish_reason, "end_turn")

    def _convert_usage(self, openai_usage: Dict[str, Any]) -> anthropic.types.Usage:
        """Convert OpenAI usage stats to Anthropic format"""
        return anthropic.types.Usage(
            input_tokens=openai_usage.get("prompt_tokens", 0),
            output_tokens=openai_usage.get("completion_tokens", 0),
        )
