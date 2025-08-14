import json
from typing import Any, Dict, List, Union

from .tool_converter import ToolConverter


class RequestConverter:
    """Converts Anthropic Messages API requests to OpenAI ChatCompletions API requests"""

    def __init__(self) -> None:
        self.tool_converter = ToolConverter()

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

        # Handle tool definitions
        if "tools" in anthropic_request:
            openai_request["tools"] = (
                self.tool_converter.convert_anthropic_tools_to_openai(
                    anthropic_request["tools"]
                )
            )

        # Handle tool choice parameter
        if "tool_choice" in anthropic_request:
            openai_request["tool_choice"] = (
                self.tool_converter.convert_tool_choice_to_openai(
                    anthropic_request["tool_choice"]
                )
            )

        # Convert messages with tool support
        openai_request["messages"] = self._convert_messages_with_tools(
            anthropic_request
        )

        return openai_request

    def _convert_messages_with_tools(
        self, anthropic_request: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert messages handling both regular content and tool results"""
        messages = []

        # Handle system message
        if "system" in anthropic_request:
            messages.append({"role": "system", "content": anthropic_request["system"]})

        # Process each message
        for message in anthropic_request.get("messages", []):
            if message["role"] in ["user", "assistant"]:
                converted_message = self._convert_message_content(message)
                if converted_message:
                    if isinstance(converted_message, list):
                        messages.extend(converted_message)
                    else:
                        messages.append(converted_message)

        return messages

    def _convert_message_content(
        self, message: Dict[str, Any]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        """Convert message content handling tool results and regular content"""
        role = message["role"]
        content = message["content"]

        # Handle string content (simple case)
        if isinstance(content, str):
            return {"role": role, "content": content}

        # Handle content blocks (list)
        if isinstance(content, list):
            # Check if this contains tool results (user messages)
            tool_result_blocks = [
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]

            if tool_result_blocks:
                # Convert tool result blocks to OpenAI tool messages
                tool_messages = (
                    self.tool_converter.convert_anthropic_tool_results_to_openai(
                        content
                    )
                )

                # Handle any text content alongside tool results
                text_blocks = [
                    block
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                regular_messages = []

                if text_blocks:
                    text_content = " ".join(block["text"] for block in text_blocks)
                    if text_content.strip():
                        regular_messages.append({"role": role, "content": text_content})

                # Return both regular messages and tool messages
                return regular_messages + tool_messages

            # Check if this contains tool use blocks (assistant messages)
            tool_use_blocks = [
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_use"
            ]

            if tool_use_blocks:
                # Handle assistant messages with tool calls
                message_dict = {"role": role}

                # Get text content
                text_blocks = [
                    block
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if text_blocks:
                    text_content = " ".join(block["text"] for block in text_blocks)
                    message_dict["content"] = (
                        text_content if text_content.strip() else None
                    )
                else:
                    message_dict["content"] = None

                # Convert tool use blocks to OpenAI tool calls
                tool_calls = []
                for block in tool_use_blocks:
                    tool_call = {
                        "id": self.tool_converter._convert_tool_use_id_to_call_id(
                            block["id"]
                        ),
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    }
                    tool_calls.append(tool_call)

                message_dict["tool_calls"] = tool_calls
                return message_dict  # type: ignore
            else:
                # Handle other content blocks (like text blocks only)
                text_blocks = [
                    block
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if text_blocks:
                    text_content = " ".join(block["text"] for block in text_blocks)
                    if text_content.strip():
                        return {"role": role, "content": text_content}

        return None
