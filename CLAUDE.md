# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Anthropic-OpenAI bridge library that converts between Anthropic Messages API and OpenAI ChatCompletions API formats. The bridge allows using Anthropic's Python SDK format while communicating with OpenAI-compatible LLM services, particularly for internet-disconnected networks.

## Architecture

**Core Components**:
- **`AnthropicOpenAIBridge`** (`src/anthropic_openai_bridge/bridge.py`): Main orchestrator
- **`RequestConverter`** (`src/anthropic_openai_bridge/converters/request_converter.py`): Anthropic → OpenAI request conversion
- **`ResponseConverter`** (`src/anthropic_openai_bridge/converters/response_converter.py`): OpenAI → Anthropic response conversion
- **`ToolConverter`** (`src/anthropic_openai_bridge/converters/tool_converter.py`): Tool/function calling format conversion
- **`OpenAIClientWrapper`** (`src/anthropic_openai_bridge/client/openai_client.py`): OpenAI API communication
- **`ConfigManager`** (`src/anthropic_openai_bridge/config/config_manager.py`): Configuration and API key management

**Data Flow**: Anthropic request dict → RequestConverter → OpenAI API → ResponseConverter → `anthropic.types.Message` object

## Critical Implementation Details

**Return Type**: The bridge returns `anthropic.types.Message` objects, NOT dictionaries:
```python
response = bridge.send_message(request)  # Returns anthropic.types.Message object
print(response.content[0].text)          # ✅ Object notation
print(response["content"][0]["text"])    # ❌ Dictionary access fails
```

**Model Name Passthrough**: Model names are passed unchanged - no mapping occurs. Use any model identifier your OpenAI-compatible service expects.

**Tool Calling**: Full bidirectional conversion between Anthropic and OpenAI tool calling formats, including multi-turn conversations and tool choice parameters.

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

# Run tool calling specific tests
python -m pytest tests/unit/test_tool_converter.py -v
python -m pytest tests/integration/test_tool_calling_integration.py -v

# Run all tool-related tests
python -m pytest -k "tool" -v

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

# Run single test class
python -m pytest tests/unit/test_bridge.py::TestAnthropicOpenAIBridge -v
```

## Implementation Status

**Phase 1 & 2 Complete**: Full bridge functionality with tool calling support
- ✅ Bidirectional API format conversion (Anthropic ↔ OpenAI)
- ✅ Tool calling conversion (tools ↔ functions, tool_use ↔ function_call)
- ✅ Multi-turn conversations with tool results
- ✅ Custom httpx client support for network security
- ✅ Returns proper `anthropic.types.Message` objects (not dictionaries)

## Testing Strategy

**Test Structure**:
- `tests/unit/`: Component isolation tests (converters, config, client)
- `tests/integration/`: End-to-end bridge functionality tests
- `tests/fixtures/`: JSON fixtures for consistent test data across tool calling and basic scenarios

**Key Test Files**:
- `test_tool_converter.py`: Tool format conversion logic
- `test_request_converter_tools.py` / `test_response_converter_tools.py`: Tool calling integration in converters
- `test_tool_calling_integration.py`: Full tool calling workflows