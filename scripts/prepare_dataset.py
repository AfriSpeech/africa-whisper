#!/usr/bin/env python3
"""
Standardized Data Preparation Script for Africa-Whisper.

All datasets used to pre-train or fine-tune Africa-Whisper must be prepared using
the exact same orthographic normalization engine: Africa-G2P (universal mode).

This maps language-specific characters, diacritics, non-Latin scripts (e.g. Ge'ez, Arabic),
and tone marks into a universal 26-character Latin representation (a-z) with preserved
word boundaries (single spaces) and stripped punctuation.

Requirements:
    pip install africa-g2p librosa soundfile datasets

Usage:
    # 1. Normalize a single string:
    from prepare_dataset import universalize_text
    clean_text = universalize_text("Onyankopɔn, me din de Kwame! Ɔyɛ kɛse.", lang="twi")
    # -> 'onyankopon me din de kwame oye kese'

    # 2. Process a dataset from Hugging Face or CSV/JSON:
    python prepare_dataset.py --dataset_name "google/fleurs" --subset "yo_ng" --output_dir ./prepared_data
"""

import re
import unicodedata
from functools import lru_cache
from typing import Optional

try:
    from africa_g2p import GraphemeConverter, UNIVERSAL
    from africa_g2p.convert import universal_fallback
except ImportError:
    raise ImportError(
        "africa-g2p is required for Africa-Whisper data preparation.\n"
        "Install it via: pip install git+https://github.com/AfriSpeech/africa-g2p.git"
    )

_NON_AZ_SPACE = re.compile(r"[^a-z\s]+")
_MULTISPACE = re.compile(r"\s+")

# Common language code aliases to map standard ISO 639-3 or regional codes to africa-g2p rule sets
RULE_ALIASES: dict[str, str] = {
    "aka": "twi",      # Akan -> Twi
    "swa": "swh",      # Swahili macro -> Swahili
    "swc": "swh",      # Congo Swahili -> Swahili
    "pcm": "kri",      # Nigerian Pidgin -> Krio
    "wes": "kri",      # West African Pidgin -> Krio
    "kri": "kri",      # Krio
    "dnj": "daf",      # Dan
    "kea": "kea",      # Kabuverdianu
    "mfe": "mfe",      # Mauritian Creole
    "rcf": "mfe",      # Reunion Creole -> Mauritian Creole
    "crs": "mfe",      # Seychelles Creole -> Mauritian Creole
    "aao": "arq",      # Algerian Saharan Arabic -> Algerian
    "aec": "arz",      # Saidi Arabic -> Egyptian
    "aju": "ary",      # Judeo-Moroccan Arabic -> Moroccan
    "apd": "apd",      # Sudanese Arabic
    "arq": "arq",      # Algerian Arabic
    "ary": "ary",      # Moroccan Arabic
    "arz": "arz",      # Egyptian Arabic
    "yo_ng": "yor",    # FLEURS Yoruba
    "ha_ng": "hau",    # FLEURS Hausa
    "ig_ng": "ibo",    # FLEURS Igbo
    "sn_zw": "sna",    # FLEURS Shona
    "wo_sn": "wol",    # FLEURS Wolof
}


@lru_cache(maxsize=1024)
def _get_converter(lang: str) -> Optional[GraphemeConverter]:
    try:
        return GraphemeConverter(lang, UNIVERSAL)
    except Exception:
        return None


def _fallback_text(text: str) -> str:
    parts = []
    for ch in text:
        mapped = universal_fallback(ch)
        parts.append(mapped if mapped else ch)
    s = "".join(parts)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def universalize_text(text: str, lang: str) -> str:
    """
    Converts any African speech transcript into universal lowercase Latin text (a-z)
    with natural word boundaries (single spaces) and zero punctuation.

    Args:
        text (str): Raw transcript in surface orthography (Latin, Ge'ez, Arabic, etc.)
        lang (str): ISO 639-3 code or dataset language identifier (e.g. 'twi', 'yor', 'hau', 'wol')

    Returns:
        str: Universalized string containing only characters in [a-z ]
    """
    if not text or not isinstance(text, str):
        return ""

    code = RULE_ALIASES.get(lang.lower(), lang.lower())
    conv = _get_converter(code) or _get_converter(lang.lower())

    if conv is not None:
        try:
            out = conv.convert(text, sep="", lower=True)
        except Exception:
            out = _fallback_text(text)
    else:
        out = _fallback_text(text)

    # Clean characters and ensure single spacing
    out = _NON_AZ_SPACE.sub(" ", out.lower())
    return _MULTISPACE.sub(" ", out).strip()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test or batch-convert text using Africa-G2P universalization")
    parser.add_argument("--text", type=str, default="Onyankopɔn, me din de Kwame! Ɔyɛ kɛse.", help="Sample text")
    parser.add_argument("--lang", type=str, default="twi", help="Language ISO 639-3 code")
    args = parser.parse_args()

    result = universalize_text(args.text, args.lang)
    print(f"Original : '{args.text}'")
    print(f"Universal: '{result}'")
