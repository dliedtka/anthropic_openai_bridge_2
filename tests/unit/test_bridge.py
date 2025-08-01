import pytest
from unittest.mock import Mock, patch
from anthropic_openai_bridge.bridge import AnthropicOpenAIBridge
from anthropic_openai_bridge.config.config_manager import ConfigManager


class TestAnthropicOpenAIBridge:
    def test_init_with_default_config(self):
        """Test bridge initialization with default configuration"""
        with patch.object(ConfigManager, '__init__', return_value=None) as mock_config_init:
            mock_config = Mock()
            mock_config_init.return_value = None
            
            with patch('anthropic_openai_bridge.bridge.ConfigManager', return_value=mock_config):
                with patch('anthropic_openai_bridge.bridge.OpenAIClientWrapper'), \
                     patch('anthropic_openai_bridge.bridge.RequestConverter'), \
                     patch('anthropic_openai_bridge.bridge.ResponseConverter'):
                    
                    bridge = AnthropicOpenAIBridge()
                    
                    assert bridge.config == mock_config
    
    def test_init_with_custom_config_manager(self):
        """Test bridge initialization with custom config manager"""
        custom_config = Mock()
        
        with patch('anthropic_openai_bridge.bridge.OpenAIClientWrapper'), \
             patch('anthropic_openai_bridge.bridge.RequestConverter'), \
             patch('anthropic_openai_bridge.bridge.ResponseConverter'):
            
            bridge = AnthropicOpenAIBridge(config_manager=custom_config)
            
            assert bridge.config == custom_config
    
    def test_init_with_custom_openai_api_key(self):
        """Test bridge initialization with custom OpenAI API key"""
        with patch('anthropic_openai_bridge.bridge.ConfigManager') as mock_config_class, \
             patch('anthropic_openai_bridge.bridge.OpenAIClientWrapper'), \
             patch('anthropic_openai_bridge.bridge.RequestConverter'), \
             patch('anthropic_openai_bridge.bridge.ResponseConverter'):
            
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            
            bridge = AnthropicOpenAIBridge(openai_api_key="custom_key")
            
            # Verify ConfigManager was created with custom API key
            mock_config_class.assert_called_once_with(
                openai_api_key="custom_key",
                openai_base_url=None,
                httpx_client=None
            )
            assert bridge.config == mock_config
    
    def test_init_with_custom_openai_base_url(self):
        """Test bridge initialization with custom OpenAI base URL"""
        with patch('anthropic_openai_bridge.bridge.ConfigManager') as mock_config_class, \
             patch('anthropic_openai_bridge.bridge.OpenAIClientWrapper'), \
             patch('anthropic_openai_bridge.bridge.RequestConverter'), \
             patch('anthropic_openai_bridge.bridge.ResponseConverter'):
            
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            
            bridge = AnthropicOpenAIBridge(openai_base_url="https://custom.endpoint.com")
            
            # Verify ConfigManager was created with custom base URL
            mock_config_class.assert_called_once_with(
                openai_api_key=None,
                openai_base_url="https://custom.endpoint.com",
                httpx_client=None
            )
            assert bridge.config == mock_config
    
    def test_init_with_custom_httpx_client(self):
        """Test bridge initialization with custom httpx client"""
        mock_httpx_client = Mock()
        
        with patch('anthropic_openai_bridge.bridge.ConfigManager') as mock_config_class, \
             patch('anthropic_openai_bridge.bridge.OpenAIClientWrapper'), \
             patch('anthropic_openai_bridge.bridge.RequestConverter'), \
             patch('anthropic_openai_bridge.bridge.ResponseConverter'):
            
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            
            bridge = AnthropicOpenAIBridge(httpx_client=mock_httpx_client)
            
            # Verify ConfigManager was created with custom httpx client
            mock_config_class.assert_called_once_with(
                openai_api_key=None,
                openai_base_url=None,
                httpx_client=mock_httpx_client
            )
            assert bridge.config == mock_config
    
    def test_init_with_all_custom_parameters(self):
        """Test bridge initialization with all custom parameters"""
        mock_httpx_client = Mock()
        
        with patch('anthropic_openai_bridge.bridge.ConfigManager') as mock_config_class, \
             patch('anthropic_openai_bridge.bridge.OpenAIClientWrapper'), \
             patch('anthropic_openai_bridge.bridge.RequestConverter'), \
             patch('anthropic_openai_bridge.bridge.ResponseConverter'):
            
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            
            bridge = AnthropicOpenAIBridge(
                openai_api_key="custom_key",
                openai_base_url="https://custom.endpoint.com",
                httpx_client=mock_httpx_client
            )
            
            # Verify ConfigManager was created with all custom parameters
            mock_config_class.assert_called_once_with(
                openai_api_key="custom_key",
                openai_base_url="https://custom.endpoint.com",
                httpx_client=mock_httpx_client
            )
            assert bridge.config == mock_config
    
    def test_init_custom_params_with_existing_config_manager(self):
        """Test that custom parameters are ignored when config_manager is provided"""
        custom_config = Mock()
        
        with patch('anthropic_openai_bridge.bridge.OpenAIClientWrapper'), \
             patch('anthropic_openai_bridge.bridge.RequestConverter'), \
             patch('anthropic_openai_bridge.bridge.ResponseConverter'):
            
            bridge = AnthropicOpenAIBridge(
                config_manager=custom_config,
                openai_api_key="should_be_ignored",
                openai_base_url="should_be_ignored"
            )
            
            # Should use the provided config_manager, not create a new one
            assert bridge.config == custom_config
    
    def test_send_message_flow(self):
        """Test the complete send_message flow"""
        with patch('anthropic_openai_bridge.bridge.ConfigManager'), \
             patch('anthropic_openai_bridge.bridge.OpenAIClientWrapper') as mock_client_class, \
             patch('anthropic_openai_bridge.bridge.RequestConverter') as mock_req_converter_class, \
             patch('anthropic_openai_bridge.bridge.ResponseConverter') as mock_resp_converter_class:
            
            # Set up mocks
            mock_openai_client = Mock()
            mock_client_class.return_value = mock_openai_client
            
            mock_req_converter = Mock()
            mock_req_converter_class.return_value = mock_req_converter
            
            mock_resp_converter = Mock()
            mock_resp_converter_class.return_value = mock_resp_converter
            
            # Set up method returns
            anthropic_request = {"model": "test", "messages": []}
            openai_request = {"model": "test", "messages": []}
            openai_response = {"choices": [{"message": {"content": "response"}}]}
            anthropic_response = {"content": [{"text": "response"}]}
            
            mock_req_converter.convert.return_value = openai_request
            mock_openai_client.create_chat_completion.return_value = openai_response
            mock_resp_converter.convert.return_value = anthropic_response
            
            bridge = AnthropicOpenAIBridge()
            result = bridge.send_message(anthropic_request)
            
            # Verify the flow
            mock_req_converter.convert.assert_called_once_with(anthropic_request)
            mock_openai_client.create_chat_completion.assert_called_once_with(openai_request)
            mock_resp_converter.convert.assert_called_once_with(openai_response)
            assert result == anthropic_response