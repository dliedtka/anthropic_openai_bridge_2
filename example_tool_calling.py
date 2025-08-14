#!/usr/bin/env python3
"""
Tool Calling Example for Anthropic-OpenAI Bridge

This script demonstrates how to use the bridge with tool calling capabilities.
It shows both single and multiple tool usage patterns.
"""

import json
from anthropic_openai_bridge import AnthropicOpenAIBridge


def execute_weather_tool(location: str, unit: str = "fahrenheit") -> str:
    """Mock weather tool implementation"""
    # In a real implementation, you would call a weather API
    mock_weather = {
        "San Francisco, CA": "68°F, partly cloudy" if unit == "fahrenheit" else "20°C, partly cloudy",
        "Boston, MA": "42°F, snow showers" if unit == "fahrenheit" else "6°C, snow showers", 
        "New York, NY": "55°F, clear skies" if unit == "fahrenheit" else "13°C, clear skies"
    }
    return mock_weather.get(location, f"Weather data unavailable for {location}")


def execute_calculator_tool(expression: str) -> str:
    """Mock calculator tool implementation"""
    try:
        # In production, you'd want safer evaluation
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating {expression}: {e}"


def main():
    """Demonstrate tool calling functionality"""
    
    # Initialize the bridge
    try:
        bridge = AnthropicOpenAIBridge()
        print("✓ Bridge initialized successfully")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("Make sure you have a .env file with OPENAI_API_KEY and ANTHROPIC_API_KEY")
        return
    
    # Example 1: Single Tool Usage
    print("\n--- Example 1: Single Tool Usage ---")
    
    weather_request = {
        "model": "your_model_name",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in San Francisco?"
            }
        ]
    }
    
    try:
        # First request - model decides to use tool
        response = bridge.send_message(weather_request)
        print(f"Model response: {response.content[0].text if response.content else 'No text content'}")
        print(f"Stop reason: {response.stop_reason}")
        
        if response.stop_reason == "tool_use":
            # Find tool use blocks
            tool_uses = [block for block in response.content if hasattr(block, 'type') and block.type == "tool_use"]
            
            if tool_uses:
                tool_use = tool_uses[0]
                print(f"Tool requested: {tool_use.name}")
                print(f"Tool input: {tool_use.input}")
                
                # Execute the tool
                tool_result = execute_weather_tool(
                    tool_use.input["location"],
                    tool_use.input.get("unit", "fahrenheit")
                )
                print(f"Tool result: {tool_result}")
                
                # Continue conversation with tool result
                follow_up_request = {
                    "model": "your_model_name",
                    "max_tokens": 1024,
                    "tools": weather_request["tools"],
                    "messages": [
                        *weather_request["messages"],
                        {
                            "role": "assistant",
                            "content": response.content
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use.id,
                                    "content": tool_result
                                }
                            ]
                        }
                    ]
                }
                
                final_response = bridge.send_message(follow_up_request)
                print(f"Final response: {final_response.content[0].text}")
                
    except Exception as e:
        print(f"Error in single tool example: {e}")
    
    # Example 2: Multiple Tools
    print("\n--- Example 2: Multiple Tools Usage ---")
    
    multi_tool_request = {
        "model": "your_model_name", 
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Boston and what's 15 multiplied by 8?"
            }
        ]
    }
    
    try:
        response = bridge.send_message(multi_tool_request)
        print(f"Model response: {response.content[0].text if response.content else 'No text content'}")
        
        if response.stop_reason == "tool_use":
            # Handle multiple tool uses
            tool_uses = [block for block in response.content if hasattr(block, 'type') and block.type == "tool_use"]
            
            tool_results = []
            for tool_use in tool_uses:
                print(f"Using tool: {tool_use.name} with input: {tool_use.input}")
                
                if tool_use.name == "get_weather":
                    result = execute_weather_tool(tool_use.input["location"])
                elif tool_use.name == "calculate":
                    result = execute_calculator_tool(tool_use.input["expression"])
                else:
                    result = f"Unknown tool: {tool_use.name}"
                
                print(f"Tool result: {result}")
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result
                })
            
            # Continue conversation with all tool results
            follow_up_request = {
                "model": "your_model_name",
                "max_tokens": 1024,
                "tools": multi_tool_request["tools"],
                "messages": [
                    *multi_tool_request["messages"],
                    {
                        "role": "assistant",
                        "content": response.content
                    },
                    {
                        "role": "user",
                        "content": tool_results
                    }
                ]
            }
            
            final_response = bridge.send_message(follow_up_request)
            print(f"Final response: {final_response.content[0].text}")
            
    except Exception as e:
        print(f"Error in multiple tools example: {e}")
    
    # Example 3: Tool Choice Control
    print("\n--- Example 3: Tool Choice Control ---")
    
    forced_tool_request = {
        "model": "your_model_name",
        "max_tokens": 1024,
        "tool_choice": {
            "type": "tool",
            "name": "get_weather"
        },
        "tools": [
            {
                "name": "get_weather", 
                "description": "Get weather information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "get_news",
                "description": "Get news information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"}
                    },
                    "required": ["topic"]
                }
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "Tell me about San Francisco"  # Ambiguous - could be weather or news
            }
        ]
    }
    
    try:
        response = bridge.send_message(forced_tool_request)
        print(f"Model was forced to use weather tool")
        
        if response.stop_reason == "tool_use":
            tool_uses = [block for block in response.content if hasattr(block, 'type') and block.type == "tool_use"]
            if tool_uses:
                tool_use = tool_uses[0]
                print(f"Tool used: {tool_use.name} (as expected)")
                print(f"Input: {tool_use.input}")
                
    except Exception as e:
        print(f"Error in tool choice example: {e}")
    
    print("\n✓ Tool calling examples completed!")


if __name__ == "__main__":
    main()