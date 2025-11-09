from src.translator import translate_content


def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english is False
    assert translated_content == "This is a Chinese message"

def test_llm_normal_response():
    # English input should echo back unchanged and be marked as English
    is_english, translated_content = translate_content("This is an English message")
    assert is_english is True
    assert translated_content == "This is an English message"

def test_llm_gibberish_response():
    # With current implementation, unknown/gibberish is echoed back as-is and marked English
    text = "??? 🤯🤯 123 !!"
    is_english, translated_content = translate_content(text)
    assert is_english is False
    assert translated_content == "Unable to translate"
