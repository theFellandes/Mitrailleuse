from mitrailleuse.infrastructure.utils import prompt_utils

def test_find_prompt_content():
    data = {"input_text": "hi", "instructions": "be helpful"}
    user, system = prompt_utils.find_prompt_content(data, "input_text", "instructions", True)
    assert user == "hi"
    assert system == "be helpful"

def test_find_text_content():
    data = {"text": "hello"}
    assert prompt_utils.find_text_content(data) == "hello"

def test_build_openai_body():
    input_data = {"input_text": "hi", "instructions": "be helpful"}
    config = {
        "openai": {
            "prompt": "input_text",
            "system_instruction": {"is_dynamic": True, "system_prompt": "instructions"},
            "api_information": {"model": "gpt-4", "setting": {"temperature": 0.7, "max_tokens": 10}}
        }
    }
    body = prompt_utils.build_openai_body(input_data, config)
    assert "model" in body and "messages" in body

def test_build_deepl_body():
    input_data = {"text": "hello"}
    config = {"deepl": {"default_target_lang": "EN-US"}}
    body = prompt_utils.build_deepl_body(input_data, config)
    assert "text" in body and "target_lang" in body
