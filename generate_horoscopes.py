#!/usr/bin/env python3
"""
Pure daily horoscope generator – no fallback, only fresh SambaNova output.
Retries each category 3 times, fails loudly if a category can't be generated.
"""

import json
import os
import time
import sys
from datetime import datetime, timezone

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

MAX_OUTPUT_TOKENS = 350
RETRY_COUNT = 3
RETRY_DELAY = 2  # seconds between retries

# ---------- GENERATION ----------
def generate_category(rashi, category, today_str):
    """Call API with retries. Raises exception on total failure."""
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

    last_exception = None
    for attempt in range(1, RETRY_COUNT + 1):
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
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("API returned None content")
            content = content.strip()
            if not content:
                raise ValueError("API returned empty content")

            # For lucky_number, extract integer
            if category == "lucky_number":
                import re
                numbers = re.findall(r'\b\d+\b', content)
                if not numbers:
                    raise ValueError(f"No number found in response: {content}")
                return int(numbers[0])

            # For lucky_color, just first word
            if category == "lucky_color":
                return content.split()[0].capitalize()

            return content

        except Exception as e:
            last_exception = e
            print(f"  Attempt {attempt}/{RETRY_COUNT} failed: {e}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"Failed to generate {category} for {rashi} after {RETRY_COUNT} attempts. "
        f"Last error: {last_exception}"
    )

# ---------- MAIN ----------
def main():
    today_str = datetime.now().strftime("%B %d, %Y")
    today_iso = datetime.now().isoformat()

    os.makedirs("data", exist_ok=True)

    output = {
        "date": today_iso,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "rashi_horoscopes": {}
    }

    for rashi in RASHIS:
        print(f"\n===== {rashi} =====")
        rashi_data = {}
        for category in CATEGORIES:
            print(f"  {category}...", end=" ", flush=True)
            result = generate_category(rashi, category, today_str)
            rashi_data[category] = result
            print("✓")
        output["rashi_horoscopes"][rashi] = rashi_data

    # Write only after full success
    with open("data/horoscopes.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n✅ All horoscopes generated and saved successfully.")

if __name__ == "__main__":
    main()
