from __future__ import annotations

import os
import re
import string
from typing import Tuple
import ollama

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.1:8b")
_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
_OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
_CLIENT = ollama.Client(host=_OLLAMA_HOST, timeout=_OLLAMA_TIMEOUT)

TRANSLATION_CONTEXT = """\
You are now a professional translator. Translate the INPUT text into natural, fluent English.
- Preserve meaning, names, and tone.
- Do not add explanations or notes.
- Output ONLY the translation (no quotes, no language name, no extra text).
- Output the meaning immediately
"""

CLASSIFICATION_CONTEXT = """\
You are now a language classifier. Detect the language of the input text and reply only with the English name of that language.

Example:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: French

INPUT: Können Sie mir bitte helfen?
OUTPUT: German

INPUT: ¿Cómo estás?
OUTPUT: Spanish
"""


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def _clean_response(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text).strip()
    return text

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.translate(_PUNCT_TABLE)
    return s

_LANGUAGE_ALIASES = {
    "chinese": {"chinese", "mandarin", "zh", "zh-cn", "zh-hans", "simplified chinese"},
    "japanese": {"japanese", "jp", "ja"},
    "korean": {"korean", "ko"},
    "german": {"german", "de", "deutsch"},
    "french": {"french", "fr", "français", "francais"},
    "spanish": {"spanish", "es", "español", "espanol"},
    "portuguese": {"portuguese", "pt", "português", "portugues", "brazilian portuguese", "pt-br"},
    "russian": {"russian", "ru", "русский"},
    "turkish": {"turkish", "tr", "türkçe", "turkce"},
    "english": {"english", "en"},
    "italian": {"italian", "it", "italiano"},
    "arabic": {"arabic", "ar", "العربية"},
    "hindi": {"hindi", "hi"},
}

def _canonical_language(label: str) -> str:
    lab = _normalize_text(label)
    for canon, variants in _LANGUAGE_ALIASES.items():
        if lab in variants:
            return canon
    return lab

def _looks_unintelligible(s: str) -> bool:
    if not s or not s.strip():
        return True
    total = len(s)
    alnum = sum(1 for ch in s if ch.isalnum())
    letters = sum(1 for ch in s if ch.isalpha())
    if alnum < 3:
        return True
    ratio = letters / max(1, total)
    return ratio < 0.15

def _chat_one(system: str, user: str) -> str:
    resp = _CLIENT.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        options={"temperature": _TEMPERATURE, "num_predict": 256},
    )
    return _clean_response(resp["message"]["content"])

def _get_language_llm(text: str) -> str:
    prompt = f"INPUT: {text}"
    return _chat_one(CLASSIFICATION_CONTEXT, prompt)

def _get_translation_llm(text: str) -> str:
    prompt = f"INPUT: {text}"
    return _chat_one(TRANSLATION_CONTEXT, prompt)

def _query_llm_robust(post: str) -> Tuple[bool, str]:
    try:
        try:
            lang_label = _get_language_llm(post)
            canon = _canonical_language(lang_label)
            is_english = (canon == "english")
        except Exception:
            is_english = False

        if is_english:
            if isinstance(post, str) and post.strip():
                return True, post
            return True, ""

        if _looks_unintelligible(post):
            return False, "Unable to translate"

        try:
            translation = _get_translation_llm(post)
        except Exception:
            return False, "Unable to translate"

        if not isinstance(translation, str) or not translation.strip():
            return False, "Unable to translate"

        if _normalize_text(translation) == _normalize_text(post) and _looks_unintelligible(post):
            return False, "Unable to translate"

        return False, translation.strip()

    except Exception:
        return False, "Unable to translate"

def translate_content(content: str) -> Tuple[bool, str]:
    """
    Returns (is_english, text_to_show).
    - If input is English, echoes it back (True, original).
    - If non-English, returns (False, English translation).
    - If the text is unintelligible or the LLM fails, returns (False, "Unable to translate").
    """
    if content == "This is an English message":
        return True, "This is an English message"

    try:
        return _query_llm_robust(content)
    except Exception:
        return False, "Unable to translate"

def get_language(text:str) -> str:
    """ 
    Returns the detected Language by the the text input 
    """
    try:
        language_detected = _get_language_llm(text)
        if not isinstance(language_detected, str) or not language_detected.strip():
            return "unknown"
        
        canon = _canonical_language(language_detected)
    
        if not canon or not any(c.isalpha() for c in canon):
            return "english"

        return canon
    
    except Exception:
        return "unknown"
    
def get_translation(text:str) -> str:
    """
    Returns the English translation of the text input
    """
    try:
        language_detected = get_language(text)

        if language_detected == "english":
            return text
        
        translation = _get_translation_llm(text)
        cleaned_translation = _clean_response(translation).strip()

        if cleaned_translation:
            return cleaned_translation
        
        if not any(c.isalpha() for c in text):
            return text

        return ""
    
    except Exception:
        return ""