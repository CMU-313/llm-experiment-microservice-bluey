from unittest.mock import patch
from src.translator import translate_content

# 1) Classifier returns nonsense; translator returns a valid string -> accept as plausible.
def test_language_nonsense_but_translation_ok():
    with patch("src.basic_llm_experiment.get_language", return_value="???"), \
         patch("src.basic_llm_experiment.get_translation", return_value="Hello there"):
        out = translate_content("Hier ist dein erstes Beispiel.")
        assert out == (False, "Hello there")

# 2) Classifier raises -> assume non-English but translator fails -> fallback.
def test_classifier_raises_then_fallback():
    with patch("src.basic_llm_experiment.get_language", side_effect=RuntimeError("down")), \
         patch("src.basic_llm_experiment.get_translation", return_value=""):
        out = translate_content("Hier ist dein erstes Beispiel.")
        assert out == (False, "Unable to translate")

# 3) Translator returns EMPTY -> fallback.
def test_translation_empty_string():
    with patch("src.basic_llm_experiment.get_language", return_value="German"), \
         patch("src.basic_llm_experiment.get_translation", return_value="   "):
        out = translate_content("Hier ist dein erstes Beispiel.")
        assert out == (False, "Unable to translate")

# 4) Translator raises -> fallback.
def test_translation_raises_exception():
    with patch("src.basic_llm_experiment.get_language", return_value="German"), \
         patch("src.basic_llm_experiment.get_translation", side_effect=RuntimeError("boom")):
        out = translate_content("Hier ist dein erstes Beispiel.")
        assert out == (False, "Unable to translate")

# 5) English input -> echo original. (translator never called)
def test_english_echo_path():
    with patch("src.basic_llm_experiment.get_language", return_value="English"):
        out = translate_content("Hello there!")
        assert out == (True, "Hello there!")

# 6) Nonsense input always safely rejected
def test_unintelligible_input_short_noise():
    with patch("src.basic_llm_experiment.get_language", return_value="Spanish"):
        out = translate_content("??? 🤯🤯 123 !!")
        assert out == (False, "Unable to translate")
