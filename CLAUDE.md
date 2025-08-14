# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Anthropic-OpenAI bridge library that enables conversion between Anthropic Messages API requests and OpenAI ChatCompletions API requests. The bridge allows users to use the Anthropic Python SDK format while communicating with an OpenAI-compatible LLM service on an internet-disconnected network.

## Architecture

The bridge is implemented with a clean separation of concerns:

- **`AnthropicOpenAIBridge`** (`src/anthropic_openai_bridge/bridge.py`): Main orchestrator class that coordinates the conversion process
- **`RequestConverter`** (`src/anthropic_openai_bridge/converters/request_converter.py`): Converts Anthropic Messages API requests to OpenAI ChatCompletions format
- **`ResponseConverter`** (`src/anthropic_openai_bridge/converters/response_converter.py`): Converts OpenAI responses back to Anthropic Messages format
- **`OpenAIClientWrapper`** (`src/anthropic_openai_bridge/client/openai_client.py`): Handles OpenAI API communication
- **`ConfigManager`** (`src/anthropic_openai_bridge/config/config_manager.py`): Manages configuration and API keys from `.env` files

### Flow
1. Takes Anthropic Messages API request dictionaries  
2. `RequestConverter` transforms to OpenAI ChatCompletions format
3. `OpenAIClientWrapper` submits to OpenAI-compatible service
4. `ResponseConverter` transforms OpenAI response back to Anthropic Message objects
5. Returns `anthropic.types.Message` object (not dictionary)

## Key Implementation Details

### Model Name Passthrough
Model names are passed through unchanged - no mapping or conversion occurs. This allows use of custom model names, OpenAI model names, or any identifier the target service expects.

### Parameter Mapping
- `max_tokens` → `max_completion_tokens` (OpenAI's preferred parameter)
- `system` parameter → system message in `messages` array
- `temperature` passes through unchanged
- Messages array structure remains compatible

### API Format Differences Handled
- **Authentication**: Manages different header formats (`x-api-key` vs `Authorization: Bearer`)
- **System Messages**: Anthropic's top-level `system` parameter converts to OpenAI's system message in messages array
- **Parameter Names**: Handles `max_tokens` vs `max_completion_tokens` difference
- **Response Structure**: Converts OpenAI response dictionaries to proper `anthropic.types.Message` objects
- **Type Safety**: Returns structured Anthropic SDK objects with full type hints

## Return Types and Object Structure

**Important**: The bridge returns proper Anthropic SDK objects, not dictionaries.

```python
response = bridge.send_message(request)
print(type(response))  # <class 'anthropic.types.message.Message'>

# Access properties using object notation (not dictionary keys):
print(response.content[0].text)    # ✅ Correct
print(response.usage.input_tokens) # ✅ Correct
print(response["content"][0]["text"])  # ❌ Wrong - this is a dict approach
```

**Object Structure**:
- `response`: `anthropic.types.Message`
- `response.content[0]`: `anthropic.types.TextBlock`
- `response.usage`: `anthropic.types.Usage`

## Environment Configuration

### Option 1: Environment File
Create `.env` file with:
- `OPENAI_API_KEY`: For OpenAI-compatible API access (required)
- `ANTHROPIC_API_KEY`: For reference/testing (optional)
- `OPENAI_BASE_URL`: Custom endpoint URL (optional, defaults to OpenAI)

### Option 2: Programmatic Configuration (Recommended for Security)
For network security requirements, use direct parameter configuration:

```python
import httpx
from anthropic_openai_bridge import AnthropicOpenAIBridge

# Custom httpx client for security requirements
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
```

This approach bypasses `.env` files and allows full control over HTTP client configuration.

## Development Commands

**Setup**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

**Testing**:
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/unit/test_request_converter.py -v
python -m pytest tests/integration/ -v

# Run single test file
python -m pytest tests/unit/test_bridge.py -v

# Quick test run
python -m pytest tests/ -q

# Run tests with coverage (requires: pip install pytest-cov)
python -m pytest tests/ --cov=src/anthropic_openai_bridge --cov-report=html
```

**Code Quality**:
```bash
# Format code (line length 88, Python 3.8+ target)
black src/ tests/

# Sort imports (black-compatible profile)
isort src/ tests/

# Type checking (strict mode: disallow_untyped_defs=true)
mypy src/

# Run all code quality checks
black src/ tests/ && isort src/ tests/ && mypy src/
```

**Build & Distribution**:
```bash
# Build package (requires build: pip install build)
python -m build

# Install locally in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

**Usage**:
```bash
# Run example script (requires .env with API keys)
python example_usage.py
```

## Current Implementation Status

**Phase 1 - Complete**: Basic conversation support implemented and tested
- ✅ Anthropic Messages request → OpenAI ChatCompletions request conversion
- ✅ OpenAI-compatible service communication 
- ✅ OpenAI ChatCompletions response → Anthropic Messages object conversion
- ✅ Returns proper `anthropic.types.Message` objects (not dictionaries)
- ✅ System message handling
- ✅ Multi-turn conversations
- ✅ Model name passthrough
- ✅ Parameter mapping and validation
- ✅ Comprehensive test suite with type safety

**Phase 2 - Planned**: Tool calling support
- 🔄 Map Anthropic tool calling format to OpenAI function calling format
- 🔄 Handle tool execution responses in both directions

## Testing Strategy

The project uses pytest with comprehensive test coverage:
- **Unit tests**: Test individual converters and components in isolation
- **Integration tests**: Test full bridge functionality end-to-end
- **Custom configuration tests**: Validate custom API keys, base URLs, and httpx client configuration
- **Fixture-based testing**: JSON fixtures in `tests/fixtures/` provide consistent test data
- **Parameter validation**: Tests ensure proper handling of edge cases and missing parameters
- **Network security testing**: Tests verify httpx client integration and custom configuration handling

### Running Tests with Custom Configuration
Tests support custom httpx clients and validate all configuration methods work correctly. The test suite includes scenarios for:
- Direct parameter configuration
- ConfigManager-based configuration 
- Environment file configuration
- Mixed configuration approaches