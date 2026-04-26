#!/usr/bin/env python3
"""
Daily Horoscope Generator for 12 Vedic Rashis.
Uses two Hugging Face models with fallback, retries, and quality checks.
Always writes a valid horoscopes.json, even if AI is completely down.
"""

import requests
import json
import os
import time
from datetime import datetime

# ---------- CONFIGURATION ----------
RASHIS = [
    "Mesha (Aries)", "Vrishabha (Taurus)", "Mithuna (Gemini)",
    "Karka (Cancer)", "Simha (Leo)", "Kanya (Virgo)",
    "Tula (Libra)", "Vrishchika (Scorpio)", "Dhanu (Sagittarius)",
    "Makara (Capricorn)", "Kumbha (Aquarius)", "Meena (Pisces)"
]

# Model endpoints (both free on HF Inference API)
PRIMARY_MODEL = "shahp7575/gpt2-horoscopes"          # Fast, purpose‑built
FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.2" # Slower but reliable

HF_TOKEN = os.environ["HF_TOKEN"]  # Set in GitHub Actions secret
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

MIN_HOROSCOPE_LENGTH = 100  # Reject if AI returns garbled output

# ---------- HELPER FUNCTIONS ----------

def call_model(api_url, prompt, max_retries=3, timeout=30):
    """Call a Hugging Face model with exponential backoff."""
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 250, "temperature": 0.8, "return_full_text": False}
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(api_url, headers=HEADERS, json=payload, timeout=timeout)
            if resp.status_code == 200:
                result = resp.json()
                # Handle different response shapes
                if isinstance(result, list) and "generated_text" in result[0]:
                    return result[0]["generated_text"].strip()
                elif isinstance(result, dict) and "generated_text" in result:
                    return result["generated_text"].strip()
            print(f"Attempt {attempt} failed: HTTP {resp.status_code}")
        except Exception as e:
            print(f"Attempt {attempt} error: {e}")
        time.sleep(2 ** attempt)  # 2s, 4s, 8s
    return None

def quality_check(text):
    """Return True if the horoscope is usable."""
    if not text or len(text) < MIN_HOROSCOPE_LENGTH:
        return False
    bogus_phrases = ["unable to generate", "i cannot", "as an ai", "error"]
    if any(phrase in text.lower() for phrase in bogus_phrases):
        return False
    return True

def load_previous_horoscopes():
    """Try to load horoscopes from yesterday's file (or the ultimate fallback)."""
    # 1. Current committed file (may be yesterday's)
    try:
        with open("data/horoscopes.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("rashi_horoscopes", {})
    except Exception:
        pass

    # 2. Hardcoded fallback (always present in repo)
    try:
        with open("data/fallback_horoscopes.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("rashi_horoscopes", {})
    except Exception:
        pass

    # 3. Absolute last resort – empty strings
    return {rashi: "Horoscope will be available soon." for rashi in RASHIS}

# ---------- MAIN ----------
def main():
    today_str = datetime.now().strftime("%B %d, %Y")
    today_iso = datetime.now().isoformat()

    # Skip if today's horoscopes already exist (to avoid double runs)
    try:
        with open("data/horoscopes.json", "r", encoding="utf-8") as f:
            existing = json.load(f)
            if existing.get("date", "").startswith(today_iso[:10]):
                print("Today's horoscopes already generated. Exiting.")
                return
    except FileNotFoundError:
        pass

    output = {"date": today_iso, "rashi_horoscopes": {}}
    previous_data = load_previous_horoscopes()

    for rashi in RASHIS:
        print(f"Processing {rashi}...")
        horoscope = None

        # ------- PROMPT TEMPLATES -------
        prompt_primary = (
            f"<|category|> general <|horoscope|> "
            f"Generate a detailed Vedic daily horoscope for {rashi}. "
            f"Today is {today_str}. Include predictions for career, love, "
            f"health, and a lucky tip. Keep it 150-200 words."
        )
        prompt_fallback = (
            f"[INST] You are a Vedic astrologer. Write a detailed, positive "
            f"daily horoscope for {rashi} for {today_str}. Cover career, love, "
            f"health, and a lucky tip. Speak directly to the reader. [/INST]"
        )

        # 1. Try primary model
        horoscope = call_model(
            f"https://api-inference.huggingface.co/models/{PRIMARY_MODEL}",
            prompt_primary
        )
        if quality_check(horoscope):
            output["rashi_horoscopes"][rashi] = horoscope
            continue

        # 2. Try fallback model
        print(f"Primary model failed for {rashi}. Trying fallback...")
        horoscope = call_model(
            f"https://api-inference.huggingface.co/models/{FALLBACK_MODEL}",
            prompt_fallback
        )
        if quality_check(horoscope):
            output["rashi_horoscopes"][rashi] = horoscope
            continue

        # 3. Reuse yesterday's (or hardcoded) text
        print(f"Both models failed for {rashi}. Reusing previous horoscope.")
        output["rashi_horoscopes"][rashi] = previous_data.get(
            rashi, f"{rashi} horoscope is being updated. Check back shortly."
        )

    # Write the final JSON file
    with open("data/horoscopes.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Horoscope generation complete.")

if __name__ == "__main__":
    main()
