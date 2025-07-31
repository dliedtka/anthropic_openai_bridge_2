import os
import pytest
import tempfile
from pathlib import Path

from anthropic_openai_bridge.config.config_manager import ConfigManager


class TestConfigManager:
    def setup_method(self):
        """Clear environment variables before each test"""
        self.original_env = {}
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"]:
            if key in os.environ:
                self.original_env[key] = os.environ[key]
                del os.environ[key]
    
    def teardown_method(self):
        """Restore environment variables after each test"""
        # Clear any variables that might have been set during test
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"]:
            if key in os.environ:
                del os.environ[key]
        
        # Restore original values
        for key, value in self.original_env.items():
            os.environ[key] = value

    def test_load_config_from_env_file(self):
        """Test loading configuration from a custom .env file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=test_openai_key\n")
            f.write("ANTHROPIC_API_KEY=test_anthropic_key\n")
            f.write("OPENAI_BASE_URL=https://custom.openai.endpoint\n")
            temp_env_file = f.name
        
        try:
            config = ConfigManager(env_file=temp_env_file)
            
            assert config.openai_api_key == "test_openai_key"
            assert config.anthropic_api_key == "test_anthropic_key"
            assert config.openai_base_url == "https://custom.openai.endpoint"
        finally:
            os.unlink(temp_env_file)
    
    def test_missing_openai_api_key_raises_error(self):
        """Test that missing OpenAI API key raises ValueError"""
        # Create empty env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("ANTHROPIC_API_KEY=test_anthropic_key\n")
            temp_env_file = f.name
        
        try:
            config = ConfigManager(env_file=temp_env_file)
            
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
                _ = config.openai_api_key
        finally:
            os.unlink(temp_env_file)
    
    def test_missing_anthropic_api_key_raises_error(self):
        """Test that missing Anthropic API key raises ValueError"""
        # Create env file with only OpenAI key
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=test_openai_key\n")
            temp_env_file = f.name
        
        try:
            config = ConfigManager(env_file=temp_env_file)
            
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is required"):
                _ = config.anthropic_api_key
        finally:
            os.unlink(temp_env_file)
    
    def test_optional_base_url(self):
        """Test that base URL is optional and returns None when not set"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=test_openai_key\n")
            f.write("ANTHROPIC_API_KEY=test_anthropic_key\n")
            temp_env_file = f.name
        
        try:
            config = ConfigManager(env_file=temp_env_file)
            
            assert config.openai_base_url is None
        finally:
            os.unlink(temp_env_file)