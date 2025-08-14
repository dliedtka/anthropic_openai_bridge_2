import json
import uuid
from typing import Any, Dict, List, Union

import anthropic.types


class ToolConverter:
    """Converts between Anthropic and OpenAI tool/function calling formats"""

    def convert_anthropic_tools_to_openai(
        self, anthropic_tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert Anthropic tool definitions to OpenAI function definitions"""
        openai_tools = []

        for tool in anthropic_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools

    def convert_openai_tool_calls_to_anthropic(
        self, openai_tool_calls: List[Dict[str, Any]]
    ) -> List[anthropic.types.ToolUseBlock]:
        """Convert OpenAI function calls to Anthropic tool use blocks"""
        tool_use_blocks = []

        for tool_call in openai_tool_calls:
            if tool_call["type"] == "function":
                function = tool_call["function"]

                # Parse arguments JSON string to object
                try:
                    arguments = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    # If JSON parsing fails, use empty dict
                    arguments = {}

                # Convert OpenAI call ID to Anthropic format
                anthropic_id = self._convert_tool_call_id(tool_call["id"])

                tool_use_block = anthropic.types.ToolUseBlock(
                    type="tool_use",
                    id=anthropic_id,
                    name=function["name"],
                    input=arguments,
                )
                tool_use_blocks.append(tool_use_block)

        return tool_use_blocks

    def convert_anthropic_tool_results_to_openai(
        self, anthropic_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert Anthropic tool_result content blocks to OpenAI tool role messages"""
        tool_messages = []

        for content_block in anthropic_content:
            if content_block.get("type") == "tool_result":
                tool_message = {
                    "role": "tool",
                    "tool_call_id": self._convert_tool_use_id_to_call_id(
                        content_block["tool_use_id"]
                    ),
                    "name": content_block.get("name", "unknown"),
                    "content": content_block["content"],
                }
                tool_messages.append(tool_message)

        return tool_messages

    def convert_tool_choice_to_openai(
        self, anthropic_tool_choice: Union[str, Dict[str, Any]]
    ) -> Union[str, Dict[str, Any]]:
        """Convert Anthropic tool_choice to OpenAI tool_choice format"""
        if isinstance(anthropic_tool_choice, str):
            if anthropic_tool_choice == "auto":
                return "auto"
            elif anthropic_tool_choice == "any":
                return "required"
            else:
                return anthropic_tool_choice
        elif (
            isinstance(anthropic_tool_choice, dict) and "type" in anthropic_tool_choice
        ):
            if anthropic_tool_choice["type"] == "tool":
                return {
                    "type": "function",
                    "function": {"name": anthropic_tool_choice["name"]},
                }

        return anthropic_tool_choice

    def _convert_tool_call_id(self, openai_call_id: str) -> str:
        """Convert OpenAI call ID to Anthropic tool use ID format"""
        # Extract meaningful part from OpenAI ID and create Anthropic-style ID
        if openai_call_id.startswith("call_"):
            suffix = openai_call_id[5:]  # Remove "call_" prefix
        else:
            suffix = openai_call_id

        # Ensure it fits Anthropic format (toolu_ prefix)
        return f"toolu_{suffix}"

    def _convert_tool_use_id_to_call_id(self, anthropic_tool_use_id: str) -> str:
        """Convert Anthropic tool use ID to OpenAI call ID format"""
        # Extract meaningful part from Anthropic ID and create OpenAI-style ID
        if anthropic_tool_use_id.startswith("toolu_"):
            suffix = anthropic_tool_use_id[6:]  # Remove "toolu_" prefix
        else:
            suffix = anthropic_tool_use_id

        # Ensure it fits OpenAI format (call_ prefix)
        return f"call_{suffix}"

    def generate_tool_call_id(self) -> str:
        """Generate a new unique tool call ID in OpenAI format"""
        return f"call_{uuid.uuid4().hex[:8]}"

    def generate_tool_use_id(self) -> str:
        """Generate a new unique tool use ID in Anthropic format"""
        return f"toolu_{uuid.uuid4().hex[:12]}"
