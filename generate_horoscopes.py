#!/usr/bin/env python3
"""
Daily Horoscope Generator using SambaNova's GPT-OSS-120B model.
Stays under 300 token limit per horoscope, uses yesterday's data as fallback.
"""

import json
import os
from datetime import datetime

# ---------- CONFIGURATION ----------
RASHIS = [
    "Mesha (Aries)", "Vrishabha (Taurus)", "Mithuna (Gemini)",
    "Karka (Cancer)", "Simha (Leo)", "Kanya (Virgo)",
    "Tula (Libra)", "Vrishchika (Scorpio)", "Dhanu (Sagittarius)",
    "Makara (Capricorn)", "Kumbha (Aquarius)", "Meena (Pisces)"
]

# SambaNova API configuration (using OpenAI-compatible client)
from sambanova import SambaNova
API_KEY = os.environ["SAMBANOVA_API_KEY"]  # Set in GitHub Actions secret
BASE_URL = "https://api.sambanova.ai/v1"

client = SambaNova(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# Token limit (strictly 300 tokens, tune prompt accordingly)
MAX_OUTPUT_TOKENS = 300

# ---------- HOROSCOPE GENERATION ----------

def generate_horoscope(rashi, today_str):
    """Generate a single horoscope using SambaNova API. Returns text or None on failure."""
    prompt = (
        f"<|category|> general <|horoscope|> "
        f"Generate a detailed Vedic daily horoscope for {rashi}. "
        f"Today is {today_str}. Include predictions for career, love, "
        f"health, and a lucky tip. Keep it concise and under 300 tokens."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a skilled Vedic astrologer. Always respond in plain text, no markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            top_p=0.9,
            max_tokens=MAX_OUTPUT_TOKENS
        )
        # The response object contains choices, we extract the content
        horoscope = response.choices[0].message.content.strip()
        return horoscope
    except Exception as e:
        print(f"SambaNova API error for {rashi}: {e}")
        return None

def load_previous_horoscopes():
    """Load yesterday's horoscopes (or hardcoded fallback)."""
    # 1. Try today's file (if exists)
    try:
        with open("data/horoscopes.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("rashi_horoscopes", {})
    except Exception:
        pass
    # 2. Hardcoded fallback
    try:
        with open("data/fallback_horoscopes.json", "r", encoding="utf-8") as f:
            return json.load(f).get("rashi_horoscopes", {})
    except Exception:
        pass
    # 3. Absolute last resort
    return {r: f"Horoscope for {r} is being updated." for r in RASHIS}

# ---------- MAIN ----------

def main():
    today_str = datetime.now().strftime("%B %d, %Y")
    today_iso = datetime.now().isoformat()

    # Skip if today's horoscopes already exist
    try:
        with open("data/horoscopes.json", "r", encoding="utf-8") as f:
            if json.load(f).get("date", "").startswith(today_iso[:10]):
                print("Today's horoscopes already generated. Exiting.")
                return
    except FileNotFoundError:
        pass

    output = {"date": today_iso, "rashi_horoscopes": {}}
    previous_data = load_previous_horoscopes()

    for rashi in RASHIS:
        print(f"Generating for {rashi}...")
        horoscope = generate_horoscope(rashi, today_str)
        
        # Quality check: ensure response is not empty and has reasonable length
        if horoscope and len(horoscope) > 50:
            output["rashi_horoscopes"][rashi] = horoscope
        else:
            # Fallback to previous day
            print(f"Failed to generate for {rashi}, using previous horoscope.")
            output["rashi_horoscopes"][rashi] = previous_data.get(
                rashi, f"Horoscope for {rashi} is being updated. Check back soon."
            )

    # Write output JSON
    with open("data/horoscopes.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("Horoscope generation complete.")

if __name__ == "__main__":
    main()
