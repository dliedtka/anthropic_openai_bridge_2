# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Anthropic-OpenAI bridge library that enables conversion between Anthropic Messages API requests and OpenAI ChatCompletions API requests. The bridge allows users to use the Anthropic Python SDK while communicating with an OpenAI-compatible LLM service on an internet-disconnected network.

## Architecture

The bridge operates by:
1. Taking Anthropic Messages Python request objects
2. Converting them to OpenAI ChatCompletions Python request objects via JSON transformation
3. Submitting requests to the OpenAI-compatible service
4. Converting OpenAI ChatCompletion response objects back to Anthropic Messages response objects
5. Returning the converted response to the user

Key design decision: Uses both Anthropic and OpenAI Python libraries for robust conversion (Messages Request → JSON → ChatCompletions Request, then ChatCompletions Response → JSON → Messages Response).

## API Mappings

### Core Features to Support
- Traditional LLM conversation turns (Phase 1)
- Tool calling functionality (Phase 2)
- Bridging API differences between the two formats

### Key API Differences
- **Message Structure**: Anthropic uses `messages` array with `role` and `content`, OpenAI has similar structure but different parameter names and nesting
- **Model Parameters**: OpenAI uses `max_completion_tokens` (replacing deprecated `max_tokens`), Anthropic has different parameter names
- **Tool Calling**: Both support function calling but with different schemas and parameter structures
- **Authentication**: Anthropic uses `x-api-key` header, OpenAI uses `Authorization: Bearer` header
- **Response Format**: Different object structures and field names between the APIs

## Environment Configuration

API keys are stored in `.env`:
- `ANTHROPIC_API_KEY`: For Anthropic API access
- `OPENAI_API_KEY`: For OpenAI API access

## Documentation References

- `anthropic_messages.md`: Official Anthropic Messages API documentation
- `openai_chatcompletions.md`: OpenAI ChatCompletions API documentation  
- `project_specification.md`: Detailed project requirements and implementation approach

## Development Status

This is a greenfield project - no implementation exists yet. The codebase currently contains only documentation and specifications.

## Implementation Approach

- Target environment is an internet-disconnected network
- Previous attempt using manual request building failed on the work network
- Current approach uses official Python SDKs for better compatibility
- Focus on supporting both basic conversations and tool calling scenarios

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

# Quick test run
python -m pytest tests/ -q
```

**Code Quality**:
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

**Usage**:
```bash
# Run example script (requires .env with API keys)
python example_usage.py
```

## Phase Implementation Plan

**Phase 1**: Basic conversation support
- Convert Anthropic Messages request → OpenAI ChatCompletions request
- Submit to OpenAI-compatible service
- Convert OpenAI ChatCompletions response → Anthropic Messages response

**Phase 2**: Tool calling support
- Map Anthropic tool calling format to OpenAI function calling format
- Handle tool execution responses in both directions