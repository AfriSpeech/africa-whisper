#!/usr/bin/env python3
"""
Official LDS African Speech Dataset Builder.

Uses the churchofjesuschrist.org JSON Content API:
    /study/api/v3/language-pages/type/content?lang={lang}&uri={uri}

Directly extracts:
1. Native language verification (data-bcp47-lang == target language, dropping English fallbacks)
2. Direct studio audio URLs (.mp3 and .m4a) from data['meta']['audio']
3. Full paragraph-level native speech transcripts
4. Africa-G2P universalized Latin targets ([a-z ]+)
5. Sentence/utterance segmentation exported to 16 kHz mono FLAC Parquet files
"""

import os
import sys
import re
import io
import time
import json
import tempfile
import argparse
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import soundfile as sf
import librosa
import pyarrow as pa
import pyarrow.parquet as pq

# Import Africa-G2P universalizer
from prepare_dataset import universalize_text

API_BASE = "https://www.churchofjesuschrist.org/study/api/v3/language-pages/type/content"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

LANG_MAP = {
    "swa": ("swa", "sw", "Kiswahili"),
    "yor": ("yor", "yo", "Yoruba"),
    "twi": ("twi", "tw", "Twi (Asante)"),
    "fat": ("fat", "fat", "Fante"),
    "ibo": ("ibo", "ig", "Igbo"),
    "mlg": ("mlg", "mg", "Malagasy"),
    "efi": ("efi", "efi", "Efik"),
    "amh": ("amh", "am", "Amharic"),
    "kin": ("kin", "rw", "Kinyarwanda"),
    "lin": ("lin", "ln", "Lingala"),
}


def fetch_json(url: str, max_retries: int = 3) -> Optional[Dict]:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    return None


def get_conference_talks(conference_slug: str, query_lang: str) -> List[str]:
    url = f"{API_BASE}?lang={query_lang}&uri=/general-conference/{conference_slug}"
    data = fetch_json(url)
    if not data:
        return []

    body = data.get("content", {}).get("body", "")
    links = re.findall(r'href=\"(/study/general-conference/[^\"]+)\"', body)
    if not links:
        links = re.findall(r'href=\"([^\"]+)\"', body)

    talk_uris = []
    for l in links:
        cleaned = re.sub(r'^/study', '', l).split('?')[0]
        if any(skip in cleaned for skip in ["saturday-morning", "saturday-afternoon", "sunday-morning", "sunday-afternoon", "general-conference-leadership-session"]):
            continue
        if cleaned.startswith(f"/general-conference/{conference_slug}/"):
            talk_uris.append(cleaned)

    return sorted(list(dict.fromkeys(talk_uris)))


def fetch_talk_data(talk_uri: str, query_lang: str, expected_bcp47: str) -> Optional[Dict]:
    url = f"{API_BASE}?lang={query_lang}&uri={talk_uri}"
    data = fetch_json(url)
    if not data:
        return None

    meta = data.get("meta", {})
    page_attrs = meta.get("pageAttributes", {})
    actual_bcp47 = page_attrs.get("data-bcp47-lang", "")

    # Validation: Only genuine language pages kept; drop English fallbacks
    if actual_bcp47 != expected_bcp47 and actual_bcp47 != query_lang:
        return None

    audio_list = meta.get("audio", [])
    if not audio_list:
        return None

    media_url = audio_list[0].get("mediaUrl", "")
    if not media_url:
        return None

    body_html = data.get("content", {}).get("body", "")
    paras = re.findall(r'<p[^>]*class=\"[^\"]*para[^\"]*\"[^>]*>(.*?)</p>', body_html, re.DOTALL)
    if not paras:
        paras = re.findall(r'<p[^>]*>(.*?)</p>', body_html, re.DOTALL)

    clean_paras = []
    for p in paras:
        t = re.sub(r'<[^>]+>', '', p).strip()
        if len(t) > 20 and not any(t.startswith(w) for w in ["2020", "2021", "2022", "2023", "2024", "2025"]):
            clean_paras.append(t)

    if not clean_paras:
        return None

    title = meta.get("title", "")
    canonical = meta.get("canonicalUrl", talk_uri)
    speaker = talk_uri.split("/")[-1]
    speaker_clean = re.sub(r'^[0-9]+', '', speaker)

    return {
        "uri": talk_uri,
        "title": title,
        "speaker": speaker_clean,
        "canonical_url": f"https://www.churchofjesuschrist.org{canonical}",
        "media_url": media_url,
        "paragraphs": clean_paras,
    }


def download_audio_bytes(media_url: str) -> Optional[bytes]:
    req = urllib.request.Request(media_url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    except Exception as e:
        print(f"Failed to download audio from {media_url}: {e}")
        return None


def decode_audio_to_16k(audio_bytes: bytes, target_sr: int = 16000) -> Optional[np.ndarray]:
    """Decodes any audio container (m4a, mp3, aac, flac) to 16 kHz mono float32 array using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f_in:
        f_in.write(audio_bytes)
        in_path = f_in.name

    out_path = in_path + ".wav"
    try:
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-ar", str(target_sr),
            "-ac", "1",
            "-f", "wav",
            out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        audio, _ = sf.read(out_path)
        return audio
    except Exception as e:
        print(f"FFmpeg transcode error: {e}")
        return None
    finally:
        for p in [in_path, out_path]:
            if os.path.exists(p):
                os.unlink(p)


def segment_talk_to_utterances(
    audio_bytes: bytes,
    paragraphs: List[str],
    lang_iso: str,
    target_sr: int = 16000,
    min_dur: float = 3.0,
    max_dur: float = 16.0,
) -> List[Dict]:
    audio = decode_audio_to_16k(audio_bytes, target_sr=target_sr)
    if audio is None:
        return []

    total_duration = len(audio) / target_sr
    full_text = " ".join(paragraphs).strip()
    words = full_text.split()
    if not words or total_duration < min_dur:
        return []

    intervals = librosa.effects.split(audio, top_db=28, frame_length=2048, hop_length=512)

    chunks = []
    cur_start, cur_end = None, None
    for st_idx, en_idx in intervals:
        st_s, en_s = st_idx / target_sr, en_idx / target_sr
        if cur_start is None:
            cur_start, cur_end = st_s, en_s
        else:
            if (en_s - cur_start) <= max_dur:
                cur_end = en_s
            else:
                if (cur_end - cur_start) >= min_dur:
                    chunks.append((cur_start, cur_end))
                cur_start, cur_end = st_s, en_s
    if cur_start is not None and (cur_end - cur_start) >= min_dur:
        chunks.append((cur_start, cur_end))

    utterances = []
    total_words = len(words)
    for st, en in chunks:
        seg_dur = en - st
        w_start = int((st / total_duration) * total_words)
        w_end = max(w_start + 1, int((en / total_duration) * total_words))
        seg_text = " ".join(words[w_start:w_end]).strip()

        sub_audio = audio[int(st * target_sr) : int(en * target_sr)]
        out_buf = io.BytesIO()
        sf.write(out_buf, sub_audio, target_sr, format="FLAC")
        flac_bytes = out_buf.getvalue()

        univ_text = universalize_text(seg_text, lang_iso)

        if len(univ_text) > 3 and seg_dur >= min_dur:
            utterances.append({
                "audio": {"bytes": flac_bytes, "path": None},
                "text": univ_text,
                "transcript": seg_text,
                "duration": float(round(seg_dur, 2)),
                "language": lang_iso,
            })

    return utterances


def build_lds_corpus(
    languages: List[str],
    conferences: List[str],
    output_dir: Path,
    max_talks_per_lang: Optional[int] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Official LDS Church African Speech Dataset Builder")
    print(f"Languages   : {languages}")
    print(f"Conferences : {conferences}")
    print(f"Output Path : {output_dir}")
    print("=" * 65)

    all_data = {l: [] for l in languages}

    for conf in conferences:
        print(f"\n--- Scanning Conference: {conf} ---")
        for lang in languages:
            if lang not in LANG_MAP:
                print(f"Unknown language code: {lang}")
                continue

            query_code, expected_bcp47, lang_name = LANG_MAP[lang]
            print(f"\n[{lang_name} ({lang})] Discovering talks in {conf}...")

            talk_uris = get_conference_talks(conf, query_code)
            if not talk_uris:
                print(f"  No talks found for {lang} in {conf}.")
                continue

            print(f"  Found {len(talk_uris)} potential talks. Extracting...")
            collected_talks = 0

            for uri in talk_uris:
                if max_talks_per_lang and collected_talks >= max_talks_per_lang:
                    break

                talk_data = fetch_talk_data(uri, query_code, expected_bcp47)
                if not talk_data:
                    continue

                print(f"  -> [{talk_data['speaker']}] {talk_data['title'][:45]} ...")
                audio_bytes = download_audio_bytes(talk_data["media_url"])
                if not audio_bytes:
                    continue

                utts = segment_talk_to_utterances(audio_bytes, talk_data["paragraphs"], lang)
                for u in utts:
                    u["speaker"] = talk_data["speaker"]
                    u["source_url"] = talk_data["canonical_url"]

                all_data[lang].extend(utts)
                collected_talks += 1
                dur_m = sum(u["duration"] for u in utts) / 60
                print(f"     Extracted {len(utts)} utterances ({dur_m:.1f} mins).")

    print("\n" + "=" * 65)
    print("Dataset Summary:")
    print("=" * 65)
    for lang, utts in all_data.items():
        if not utts:
            print(f"  {lang:5s} : 0 utterances")
            continue

        tot_h = sum(u["duration"] for u in utts) / 3600
        print(f"  {lang:5s} ({LANG_MAP[lang][2]:15s}): {len(utts):5d} utterances ({tot_h:.2f} hours)")

        lang_dir = output_dir / f"language={lang}"
        lang_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(utts)
        pq.write_table(table, lang_dir / "test.parquet")
        print(f"         Saved -> {lang_dir / 'test.parquet'}")

    print("\nCompleted building LDS dataset successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and align LDS African speech dataset")
    parser.add_argument("--languages", nargs="+", default=["fat", "twi", "swa", "yor", "ibo", "mlg", "efi"], help="Language codes")
    parser.add_argument("--conferences", nargs="+", default=["2023/10", "2023/04"], help="Conference slugs (e.g. 2023/10 2023/04)")
    parser.add_argument("--max_talks", type=int, default=None, help="Max talks per language for quick runs")
    parser.add_argument("--output_dir", type=str, default="/mnt/volume_d2wey28/projects/afripa-asr/data/lds_speech_corpus", help="Output path on H200")
    args = parser.parse_args()

    build_lds_corpus(
        languages=args.languages,
        conferences=args.conferences,
        output_dir=Path(args.output_dir),
        max_talks_per_lang=args.max_talks,
    )
