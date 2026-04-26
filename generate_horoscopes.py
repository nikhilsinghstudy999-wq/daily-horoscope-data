#!/usr/bin/env python3
"""
Detailed multi‑category daily horoscope generator using SambaNova API.
Calls the model once per category per rashi, then combines into one JSON.
"""

import json
import os
import time
from datetime import datetime

from sambanova import SambaNova

# ---------- CONFIGURATION ----------
RASHIS = [
    "Mesha (Aries)", "Vrishabha (Taurus)", "Mithuna (Gemini)",
    "Karka (Cancer)", "Simha (Leo)", "Kanya (Virgo)",
    "Tula (Libra)", "Vrishchika (Scorpio)", "Dhanu (Sagittarius)",
    "Makara (Capricorn)", "Kumbha (Aquarius)", "Meena (Pisces)"
]

CATEGORIES = [
    "general", "luck", "scope", "study", "love", "travel",
    "lucky_number", "lucky_color"
]

# Category‑specific instructions added to the base prompt
CATEGORY_PROMPTS = {
    "general": "Provide a deep, detailed general prediction for today.",
    "luck": "Analyse the luck factor for today in detail. What are the hidden opportunities?",
    "scope": "Give an in‑depth scope covering all major life areas for today.",
    "study": "Offer a comprehensive forecast for students and those pursuing knowledge today.",
    "love": "Provide a thorough analysis of romantic and relationship aspects for today.",
    "travel": "Detail any travel‑related influences and advice for today.",
    "lucky_number": "Return ONLY a single lucky number (1‑100) for today. Do not include any other text.",
    "lucky_color": "Return ONLY the name of one lucky colour for today (e.g., Red). No other text."
}

API_KEY = os.environ["SAMBANOVA_API_KEY"]
BASE_URL = "https://api.sambanova.ai/v1"
client = SambaNova(api_key=API_KEY, base_url=BASE_URL)

MAX_OUTPUT_TOKENS = 350   # Enough for deep text, but still safe
DELAY_BETWEEN_CALLS = 1.0 # seconds – stays well within 60 RPM limit

# ---------- GENERATION FUNCTIONS ----------

def generate_category(rashi, category, today_str):
    """Generate content for one category of one rashi."""
    base_instruction = CATEGORY_PROMPTS[category]
    system_msg = (
        "You are a seasoned Vedic astrologer. "
        "Always answer in plain text, no markdown. "
        "Be specific, positive, and detailed."
    )
    user_prompt = (
        f"Rashi: {rashi}\nToday: {today_str}\nCategory: {category}\n"
        f"{base_instruction}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            top_p=0.9,
            max_tokens=MAX_OUTPUT_TOKENS
        )
        content = response.choices[0].message.content.strip()
        # For lucky_number, try to parse an integer
        if category == "lucky_number":
            import re
            numbers = re.findall(r'\b\d+\b', content)
            if numbers:
                return int(numbers[0])
            else:
                return None  # fallback
        # For lucky_color, just take the first word (or as returned)
        if category == "lucky_color":
            # Keep only first word if multiple
            return content.split()[0].capitalize()
        return content
    except Exception as e:
        print(f"Error generating {category} for {rashi}: {e}")
        return None

def load_previous_horoscopes():
    """Load yesterday's full data (or fallback)."""
    try:
        with open("data/horoscopes.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("rashi_horoscopes", {})
    except Exception:
        pass
    try:
        with open("data/fallback_horoscopes.json", "r", encoding="utf-8") as f:
            return json.load(f).get("rashi_horoscopes", {})
    except Exception:
        pass
    # Empty fallback
    return {r: {} for r in RASHIS}

# ---------- MAIN ----------

def main():
    today_str = datetime.now().strftime("%B %d, %Y")
    today_iso = datetime.now().isoformat()

    # Avoid redundant generation
    try:
        with open("data/horoscopes.json", "r", encoding="utf-8") as f:
            if json.load(f).get("date", "").startswith(today_iso[:10]):
                print("Today's data already generated. Exiting.")
                return
    except FileNotFoundError:
        pass

    output = {"date": today_iso, "rashi_horoscopes": {}}
    previous_data = load_previous_horoscopes()

    for rashi in RASHIS:
        print(f"\n===== {rashi} =====")
        rashi_data = {}
        for category in CATEGORIES:
            print(f"  {category}...", end=" ", flush=True)
            result = generate_category(rashi, category, today_str)
            if result is not None and (category not in ["lucky_number", "lucky_color"] or result):
                rashi_data[category] = result
                print("✓")
            else:
                # Fallback: try to reuse yesterday's value for this rashi/category
                prev_rashi = previous_data.get(rashi, {})
                fallback_val = prev_rashi.get(category, "Information not available")
                rashi_data[category] = fallback_val
                print(f"✗ (used fallback)")
            time.sleep(DELAY_BETWEEN_CALLS)  # polite rate limiting
        output["rashi_horoscopes"][rashi] = rashi_data

    with open("data/horoscopes.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\nAll horoscopes generated and saved.")

if __name__ == "__main__":
    main()
