# Anthropic-OpenAI Bridge

A Python library that enables you to use the Anthropic Messages API format while communicating with OpenAI-compatible LLM services. This bridge is particularly useful for internet-disconnected networks where you have an OpenAI-compatible API endpoint but want to use Anthropic's SDK and request format.

## Overview

The bridge operates by:
1. Taking Anthropic Messages API request objects
2. Converting them to OpenAI ChatCompletions API request objects
3. Submitting requests to your OpenAI-compatible service
4. Converting OpenAI ChatCompletion response objects back to Anthropic Messages response objects
5. Returning the converted response to you in Anthropic format

## Features

- ✅ **Complete API Format Conversion**: Seamlessly converts between Anthropic and OpenAI API formats
- ✅ **Model Name Passthrough**: Use any model name - custom models, OpenAI models, or your own naming scheme
- ✅ **System Message Support**: Anthropic `system` parameter converts to OpenAI system messages
- ✅ **Multi-turn Conversations**: Full support for conversation history
- ✅ **Parameter Mapping**: Handles differences like `max_tokens` → `max_completion_tokens`
- ✅ **Usage Statistics**: Token usage conversion between formats
- ✅ **Error Handling**: Proper error propagation and handling
- ✅ **Environment Configuration**: Secure API key management via `.env` files

## Installation

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd anthropic_openai_bridge_2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### Requirements

- Python 3.8+
- `anthropic>=0.20.0`
- `openai>=1.0.0` 
- `python-dotenv>=0.19.0`

## Configuration

Create a `.env` file in your project root:

```bash
# Required: OpenAI-compatible API endpoint
OPENAI_API_KEY=your_openai_compatible_api_key

# Required: Anthropic API key (for reference/testing)
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional: Custom OpenAI endpoint URL
OPENAI_BASE_URL=https://your-custom-openai-endpoint.com/v1
```

## Quick Start

```python
from anthropic_openai_bridge import AnthropicOpenAIBridge

# Initialize the bridge
bridge = AnthropicOpenAIBridge()

# Use Anthropic format, get Anthropic response
request = {
    "model": "your_model_name",  # Any model name - passes through unchanged
    "max_tokens": 1024,
    "messages": [
        {
            "role": "user",
            "content": "Hello! How are you?"
        }
    ]
}

response = bridge.send_message(request)
print(response["content"][0]["text"])
```

## Usage Examples

### Basic Conversation

```python
from anthropic_openai_bridge import AnthropicOpenAIBridge

bridge = AnthropicOpenAIBridge()

request = {
    "model": "our_model_1",  # Custom model name
    "max_tokens": 100,
    "messages": [
        {
            "role": "user", 
            "content": "What is the capital of France?"
        }
    ]
}

response = bridge.send_message(request)
print(f"Response: {response['content'][0]['text']}")
print(f"Usage: {response['usage']['input_tokens']} input, {response['usage']['output_tokens']} output tokens")
```

### Conversation with System Message

```python
request = {
    "model": "gpt-4-turbo",  # OpenAI-style model names work too
    "max_tokens": 150,
    "system": "You are a helpful math tutor who explains concepts clearly.",
    "messages": [
        {
            "role": "user",
            "content": "Explain the Pythagorean theorem"
        }
    ]
}

response = bridge.send_message(request)
print(f"Response: {response['content'][0]['text']}")
print(f"Stop reason: {response['stop_reason']}")
```

### Multi-turn Conversation

```python
request = {
    "model": "company_llm_v2",  # Any custom model name
    "max_tokens": 100,
    "messages": [
        {
            "role": "user",
            "content": "I'm learning Python."
        },
        {
            "role": "assistant",
            "content": "That's great! Python is a wonderful language. What would you like to know?"
        },
        {
            "role": "user", 
            "content": "How do I create a list?"
        }
    ]
}

response = bridge.send_message(request)
print(response["content"][0]["text"])
```

### Custom Configuration

The bridge supports multiple ways to configure custom OpenAI clients for secure network environments:

#### Method 1: Direct Parameters (Recommended)

```python
from anthropic_openai_bridge import AnthropicOpenAIBridge
import httpx

# For network security requirements
custom_httpx_client = httpx.Client(
    proxies="http://your-proxy:8080",
    verify="/path/to/custom/ca-cert.pem",
    timeout=30.0
)

bridge = AnthropicOpenAIBridge(
    openai_api_key="your_custom_api_key",
    openai_base_url="https://yourbaseurl.com/api",
    httpx_client=custom_httpx_client
)

response = bridge.send_message(request)
```

#### Method 2: Using ConfigManager

```python
from anthropic_openai_bridge import AnthropicOpenAIBridge
from anthropic_openai_bridge.config.config_manager import ConfigManager
import httpx

# Create custom httpx client
custom_httpx_client = httpx.Client(
    proxies="http://your-proxy:8080",
    verify="/path/to/custom/ca-cert.pem"
)

# Use custom configuration
config = ConfigManager(
    env_file="path/to/custom/.env",
    openai_api_key="your_custom_api_key",
    openai_base_url="https://yourbaseurl.com/api",
    httpx_client=custom_httpx_client
)

bridge = AnthropicOpenAIBridge(config_manager=config)
response = bridge.send_message(request)
```

#### Method 3: Environment File Only

```python
from anthropic_openai_bridge import AnthropicOpenAIBridge
from anthropic_openai_bridge.config.config_manager import ConfigManager

# Use custom .env file
config = ConfigManager(env_file="path/to/custom/.env")
bridge = AnthropicOpenAIBridge(config_manager=config)

response = bridge.send_message(request)
```

## API Reference

### AnthropicOpenAIBridge

The main bridge class that orchestrates the conversion process.

#### `__init__(config_manager=None, openai_api_key=None, openai_base_url=None, httpx_client=None)`

Initialize the bridge.

- `config_manager` (optional): Custom `ConfigManager` instance. If `None`, uses default configuration.
- `openai_api_key` (optional): Custom OpenAI API key (overrides environment variable)
- `openai_base_url` (optional): Custom OpenAI base URL (overrides environment variable) 
- `httpx_client` (optional): Custom httpx client for network security requirements

#### `send_message(anthropic_request)`

Send a message through the bridge.

- `anthropic_request` (dict): Request in Anthropic Messages API format
- Returns: Response in Anthropic Messages API format

### Request Format

The bridge accepts standard Anthropic Messages API requests:

```python
{
    "model": "your_model_name",      # Required: any model name
    "max_tokens": 1024,              # Required: maximum tokens to generate
    "messages": [...],               # Required: conversation messages
    "system": "system prompt",       # Optional: system message
    "temperature": 0.7,              # Optional: temperature parameter
    # ... other Anthropic parameters
}
```

### Response Format

Returns standard Anthropic Messages API responses:

```python
{
    "id": "msg_ABC123",
    "type": "message", 
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "The assistant's response"
        }
    ],
    "model": "your_model_name",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 10,
        "output_tokens": 25
    }
}
```

## Model Name Handling

The bridge **does not modify model names**. Whatever model name you specify in your Anthropic request will be passed through unchanged to the OpenAI API:

- ✅ `"our_model_1"` → `"our_model_1"`
- ✅ `"gpt-4-turbo"` → `"gpt-4-turbo"`  
- ✅ `"company_llm_v2"` → `"company_llm_v2"`
- ✅ Any custom model name works

This allows you to use whatever model identifier your OpenAI-compatible service expects.

## Parameter Mapping

The bridge handles key differences between the APIs:

| Anthropic | OpenAI | Notes |
|-----------|---------|-------|
| `model` | `model` | Passed through unchanged |
| `max_tokens` | `max_completion_tokens` | Parameter name change |
| `system` | `messages[0]` with `role: "system"` | System message conversion |
| `messages` | `messages` | Direct mapping |
| `temperature` | `temperature` | Passed through |

## Error Handling

The bridge propagates errors from the OpenAI API and converts them to exceptions:

```python
try:
    response = bridge.send_message(request)
except Exception as e:
    print(f"API Error: {e}")
```

Common error scenarios:
- Invalid API key
- Model not found
- Rate limiting
- Network connectivity issues

## Development

### Setup Development Environment

```bash
# Clone and setup
git clone <repo-url>
cd anthropic_openai_bridge_2
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules  
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v

# Quick test run
python -m pytest tests/ -q
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports  
isort src/ tests/

# Type checking
mypy src/
```

## Architecture

The bridge consists of several key components:

- **`RequestConverter`**: Converts Anthropic requests to OpenAI format
- **`ResponseConverter`**: Converts OpenAI responses to Anthropic format
- **`OpenAIClientWrapper`**: Handles OpenAI API communication
- **`ConfigManager`**: Manages configuration and API keys
- **`AnthropicOpenAIBridge`**: Main orchestrator class

## Use Cases

This bridge is ideal for:

- **Air-gapped/Internet-disconnected networks** with OpenAI-compatible LLM services
- **Corporate/Enterprise networks** with strict security requirements, proxy servers, and custom CA certificates
- **Migration scenarios** where you want to use Anthropic's API format with existing OpenAI infrastructure
- **Development environments** where you want to test Anthropic-format requests against local models
- **Multi-provider setups** where you need consistent API interfaces

### Network Security Features

The bridge fully supports enterprise network security requirements:

- **Custom HTTP clients**: Pass your own `httpx.Client` with custom SSL certificates, proxy settings, and timeouts
- **Proxy support**: Configure HTTP/HTTPS proxies through the httpx client
- **Custom CA certificates**: Specify custom certificate authority certificates for secure communication
- **Flexible authentication**: Support for custom API keys and endpoints

## Limitations

- **Phase 1 Implementation**: Currently supports basic conversations. Tool calling support planned for Phase 2.
- **Synchronous Only**: No async support yet (can be added in future versions)
- **Text Only**: No image/file support yet (follows Anthropic Messages API capabilities)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the test files for usage examples
- Review the source code documentation
- Create an issue in the repository

---

**Note**: This bridge uses official Anthropic and OpenAI Python SDKs for maximum compatibility and reliability.