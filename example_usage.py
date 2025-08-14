#!/usr/bin/env python3
"""
Example usage of the Anthropic-OpenAI Bridge

This script demonstrates how to use the bridge to send Anthropic-format requests
to an OpenAI-compatible API endpoint.
"""

from anthropic_openai_bridge import AnthropicOpenAIBridge


def main():
    """Example usage of the bridge"""
    
    # Initialize the bridge (reads API keys from .env file)
    try:
        bridge = AnthropicOpenAIBridge()
        print("✓ Bridge initialized successfully")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("Make sure you have a .env file with OPENAI_API_KEY and ANTHROPIC_API_KEY")
        return
    
    # Example 1: Simple conversation
    print("\n--- Example 1: Simple Conversation ---")
    simple_request = {
        "model": "our_model_1",  # Custom model name passes through unchanged
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Hello! How are you doing today?"
            }
        ]
    }
    
    try:
        response = bridge.send_message(simple_request)
        print(f"Response: {response.content[0].text}")
        print(f"Usage: {response.usage.input_tokens} input, {response.usage.output_tokens} output tokens")
    except Exception as e:
        print(f"Error in simple conversation: {e}")
    
    # Example 2: Conversation with system message
    print("\n--- Example 2: With System Message ---")
    system_request = {
        "model": "gpt-4-turbo",  # OpenAI-style model name also passes through
        "max_tokens": 150,
        "system": "You are a helpful math tutor who explains concepts clearly.",
        "messages": [
            {
                "role": "user",
                "content": "What is the Pythagorean theorem?"
            }
        ]
    }
    
    try:
        response = bridge.send_message(system_request)
        print(f"Response: {response.content[0].text}")
        print(f"Stop reason: {response.stop_reason}")
    except Exception as e:
        print(f"Error in system message example: {e}")
    
    # Example 3: Multi-turn conversation
    print("\n--- Example 3: Multi-turn Conversation ---")
    multi_turn_request = {
        "model": "custom_chat_model",  # Any custom model name works
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "I'm learning Python."
            },
            {
                "role": "assistant", 
                "content": "That's great! Python is a wonderful language to learn. What would you like to know about it?"
            },
            {
                "role": "user",
                "content": "How do I create a list?"
            }
        ]
    }
    
    try:
        response = bridge.send_message(multi_turn_request)
        print(f"Response: {response.content[0].text}")
    except Exception as e:
        print(f"Error in multi-turn example: {e}")
    
    # Example 4: Custom configuration with direct parameters
    print("\n--- Example 4: Custom Configuration (Direct Parameters) ---")
    try:
        # This is how you would configure the bridge for your network security requirements
        # In practice, you would import httpx and create your custom client
        # import httpx
        # custom_httpx_client = httpx.Client(
        #     proxies="http://your-proxy:8080",
        #     verify="/path/to/custom/ca-cert.pem",
        #     timeout=30.0
        # )
        
        bridge_custom = AnthropicOpenAIBridge(
            openai_api_key="your_custom_api_key",
            openai_base_url="https://yourbaseurl.com/api",
            # httpx_client=custom_httpx_client  # Uncomment when you have your httpx client
        )
        
        custom_request = {
            "model": "your_custom_model_name",
            "max_tokens": 50,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 2+2?"
                }
            ]
        }
        
        # This would work with your actual custom configuration
        print("✓ Custom bridge initialized successfully")
        print("  Custom API key configured")
        print("  Custom base URL: https://yourbaseurl.com/api")
        print("  (Example request would work with real configuration)")
        
    except Exception as e:
        print(f"Custom configuration example (expected without real config): {e}")
    
    # Example 5: Using ConfigManager for advanced configuration
    print("\n--- Example 5: Advanced Configuration with ConfigManager ---")
    try:
        from anthropic_openai_bridge.config.config_manager import ConfigManager
        
        # Create custom configuration manager
        custom_config = ConfigManager(
            openai_api_key="your_custom_api_key",
            openai_base_url="https://yourbaseurl.com/api"
            # httpx_client=custom_httpx_client  # Add your custom httpx client here
        )
        
        # Use the custom configuration with the bridge
        bridge_advanced = AnthropicOpenAIBridge(config_manager=custom_config)
        
        print("✓ Advanced bridge initialized successfully")
        print("  Using custom ConfigManager")
        print("  Supports all custom parameters")
        
    except Exception as e:
        print(f"Advanced configuration example: {e}")


if __name__ == "__main__":
    main()