import pytest
from unittest.mock import Mock, patch
import httpx
from anthropic_openai_bridge.client.openai_client import OpenAIClientWrapper
from anthropic_openai_bridge.config.config_manager import ConfigManager


class TestOpenAIClientWrapper:
    def test_init_with_basic_config(self):
        """Test OpenAI client initialization with basic configuration"""
        config = ConfigManager(openai_api_key="test_key")
        
        with patch('anthropic_openai_bridge.client.openai_client.openai.OpenAI') as mock_openai_class:
            wrapper = OpenAIClientWrapper(config)
            
            # Verify OpenAI client was initialized with correct parameters
            mock_openai_class.assert_called_once_with(api_key="test_key")
            assert wrapper.config == config
    
    def test_init_with_custom_base_url(self):
        """Test OpenAI client initialization with custom base URL"""
        config = ConfigManager(
            openai_api_key="test_key",
            openai_base_url="https://custom.endpoint.com"
        )
        
        with patch('anthropic_openai_bridge.client.openai_client.openai.OpenAI') as mock_openai_class:
            wrapper = OpenAIClientWrapper(config)
            
            # Verify OpenAI client was initialized with base URL
            mock_openai_class.assert_called_once_with(
                api_key="test_key",
                base_url="https://custom.endpoint.com"
            )
    
    def test_init_with_httpx_client(self):
        """Test OpenAI client initialization with custom httpx client"""
        custom_httpx_client = httpx.Client(timeout=30.0)
        config = ConfigManager(
            openai_api_key="test_key",
            httpx_client=custom_httpx_client
        )
        
        with patch('anthropic_openai_bridge.client.openai_client.openai.OpenAI') as mock_openai_class:
            wrapper = OpenAIClientWrapper(config)
            
            # Verify OpenAI client was initialized with httpx client
            mock_openai_class.assert_called_once_with(
                api_key="test_key",
                http_client=custom_httpx_client
            )
    
    def test_init_with_all_custom_parameters(self):
        """Test OpenAI client initialization with all custom parameters"""
        custom_httpx_client = httpx.Client(timeout=30.0)
        config = ConfigManager(
            openai_api_key="test_key",
            openai_base_url="https://custom.endpoint.com",
            httpx_client=custom_httpx_client
        )
        
        with patch('anthropic_openai_bridge.client.openai_client.openai.OpenAI') as mock_openai_class:
            wrapper = OpenAIClientWrapper(config)
            
            # Verify OpenAI client was initialized with all parameters
            mock_openai_class.assert_called_once_with(
                api_key="test_key",
                base_url="https://custom.endpoint.com",
                http_client=custom_httpx_client
            )
    
    def test_create_chat_completion_success(self):
        """Test successful chat completion creation"""
        config = ConfigManager(openai_api_key="test_key")
        
        with patch('anthropic_openai_bridge.client.openai_client.openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock successful response
            mock_response = Mock()
            mock_response.model_dump.return_value = {
                "id": "chatcmpl-123",
                "choices": [{"message": {"content": "Hello!"}}]
            }
            mock_client.chat.completions.create.return_value = mock_response
            
            wrapper = OpenAIClientWrapper(config)
            request = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}]}
            
            result = wrapper.create_chat_completion(request)
            
            # Verify request was made correctly
            mock_client.chat.completions.create.assert_called_once_with(**request)
            assert result == {"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello!"}}]}
    
    def test_create_chat_completion_error(self):
        """Test chat completion creation with error"""
        config = ConfigManager(openai_api_key="test_key")
        
        with patch('anthropic_openai_bridge.client.openai_client.openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock error response
            error = Exception("API Error")
            error.code = "invalid_request"
            mock_client.chat.completions.create.side_effect = error
            
            wrapper = OpenAIClientWrapper(config)
            request = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}]}
            
            result = wrapper.create_chat_completion(request)
            
            # Verify error is handled correctly
            assert "error" in result
            assert result["error"]["message"] == "API Error"
            assert result["error"]["type"] == "api_error"
            assert result["error"]["code"] == "invalid_request"
    
    def test_create_chat_completion_error_no_code(self):
        """Test chat completion creation with error that has no code attribute"""
        config = ConfigManager(openai_api_key="test_key")
        
        with patch('anthropic_openai_bridge.client.openai_client.openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock error response without code
            error = Exception("Generic Error")
            mock_client.chat.completions.create.side_effect = error
            
            wrapper = OpenAIClientWrapper(config)
            request = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}]}
            
            result = wrapper.create_chat_completion(request)
            
            # Verify error is handled correctly with default code
            assert "error" in result
            assert result["error"]["message"] == "Generic Error"
            assert result["error"]["type"] == "api_error"
            assert result["error"]["code"] == "unknown"