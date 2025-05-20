"""
Prompt Utilities  ·  May 2025
===========================
Helper functions for handling prompts and text content in various formats.
"""
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def find_prompt_content(data: Dict[str, Any], prompt_key: str, sys_prompt_key: str, is_dynamic: bool) -> Tuple[str, str]:
    """
    Find prompt content and system instruction in various formats.
    
    Args:
        data: Input data dictionary
        prompt_key: Key to look for user prompt content
        sys_prompt_key: Key to look for system prompt content
        is_dynamic: Whether to use dynamic system prompt
        
    Returns:
        Tuple of (user_content, system_content)
        
    Raises:
        ValueError: If required prompt field is not found
    """
    if "prompts" in data and isinstance(data["prompts"], list):
        # Handle nested prompts format
        if not data["prompts"]:
            raise ValueError("Empty prompts list")
        prompt_data = data["prompts"][0]  # Take first prompt
        user_content = prompt_data.get(prompt_key)
        system_content = prompt_data.get(sys_prompt_key) if is_dynamic else "You are a helpful assistant"
    else:
        # Handle direct format
        user_content = data.get(prompt_key)
        system_content = data.get(sys_prompt_key) if is_dynamic else "You are a helpful assistant"

    if not user_content:
        raise ValueError(f"Required prompt field '{prompt_key}' not found in input")

    if is_dynamic and not system_content:
        logger.warning(f"Dynamic system prompt field '{sys_prompt_key}' not found, using default")
        system_content = "You are a helpful assistant"

    return user_content, system_content

def find_text_content(data: Dict[str, Any]) -> str:
    """
    Find text content in various formats.
    
    Args:
        data: Input data dictionary
        
    Returns:
        Text content string
        
    Raises:
        ValueError: If required text field is not found
    """
    if "prompts" in data and isinstance(data["prompts"], list):
        # Handle nested prompts format
        if not data["prompts"]:
            raise ValueError("Empty prompts list")
        prompt_data = data["prompts"][0]  # Take first prompt
        text = prompt_data.get("text")
    else:
        # Handle direct format
        text = data.get("text")

    if not text:
        raise ValueError("Required 'text' field not found in input")
    return text

def build_openai_body(input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build OpenAI request body from input data and config.
    
    Args:
        input_data: Input data dictionary
        config: Configuration dictionary
        
    Returns:
        OpenAI request body dictionary
        
    Raises:
        ValueError: If required fields are not found
    """
    try:
        # Get config values
        prompt_key = config["openai"]["prompt"]
        is_dynamic = config["openai"]["system_instruction"]["is_dynamic"]
        sys_prompt_key = config["openai"]["system_instruction"]["system_prompt"]
        
        # Get content
        user_content, system_content = find_prompt_content(
            input_data, prompt_key, sys_prompt_key, is_dynamic
        )
        
        return {
            "model": config["openai"]["api_information"]["model"],
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            "temperature": config["openai"]["api_information"]["setting"]["temperature"],
            "max_tokens": config["openai"]["api_information"]["setting"]["max_tokens"]
        }
    except Exception as e:
        logger.error(f"Error building OpenAI body: {str(e)}")
        raise

def build_deepl_body(input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build DeepL request body from input data and config.
    
    Args:
        input_data: Input data dictionary
        config: Configuration dictionary
        
    Returns:
        DeepL request body dictionary
        
    Raises:
        ValueError: If required fields are not found
    """
    try:
        text = find_text_content(input_data)
        target_lang = input_data.get("target_lang", config["deepl"].get("default_target_lang", "EN-US"))
        source_lang = input_data.get("source_lang", config["deepl"].get("default_source_lang"))
        
        return {
            "text": text,
            "target_lang": target_lang,
            "source_lang": source_lang
        }
    except Exception as e:
        logger.error(f"Error building DeepL body: {str(e)}")
        raise 