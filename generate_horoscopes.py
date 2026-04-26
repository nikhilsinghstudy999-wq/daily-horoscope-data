#!/usr/bin/env python3
"""
Single‑prompt multi‑category daily horoscope generator.
Each rashi gets all 8 categories in one API call – returns structured JSON.
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

API_KEY = os.environ["SAMBANOVA_API_KEY"]
BASE_URL = "https://api.sambanova.ai/v1"
client = SambaNova(api_key=API_KEY, base_url=BASE_URL)

MAX_OUTPUT_TOKENS = 1200      # enough for 8 detailed sections
TEMPERATURE = 0.8
TOP_P = 0.9
RETRY_COUNT = 3
RETRY_DELAY = 3  # seconds

SYSTEM_PROMPT = (
    "You are a seasoned Vedic astrologer. "
    "Always respond with exactly the JSON structure requested. "
    "No markdown, no extra text, no explanations outside the JSON. "
    "The JSON keys are: general, luck, scope, study, love, travel, lucky_number, lucky_color. "
    "All values must be strings, except lucky_number which must be an integer. "
    "Make each value detailed (2-4 sentences) and specific to today."
)

# ---------- GENERATION ----------
def generate_rashi(rashi, today_str):
    """Generate all categories for one rashi in a single API call. Returns dict."""
    user_prompt = (
        f"Rashi: {rashi}\nToday: {today_str}\n"
        f"Return a JSON object (no markdown) with the following keys:\n"
        f"general (detailed prediction), luck (luck factor), scope (life areas), "
        f"study (students), love (relationships), travel (advice), "
        f"lucky_number (integer between 1-100), lucky_color (one color name)."
    )

    last_error = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-oss-120b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_OUTPUT_TOKENS
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response content")

            # Parse JSON – first try direct parse
            try:
                data = json.loads(content.strip())
            except json.JSONDecodeError:
                # If model wrapped JSON in markdown or added extra text, try to extract
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from: {content[:200]}...")

            # Validate required keys
            required_keys = {"general", "luck", "scope", "study", "love", "travel", "lucky_number", "lucky_color"}
            if not all(k in data for k in required_keys):
                raise ValueError(f"Missing keys in JSON: {data.keys()}")

            # Ensure types
            data["lucky_number"] = int(data["lucky_number"])
            data["lucky_color"] = str(data["lucky_color"])
            for key in ["general", "luck", "scope", "study", "love", "travel"]:
                data[key] = str(data[key])

            return data

        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}/{RETRY_COUNT} failed: {e}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"Failed to generate horoscope for {rashi} after {RETRY_COUNT} attempts. "
        f"Last error: {last_error}"
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
        data = generate_rashi(rashi, today_str)
        output["rashi_horoscopes"][rashi] = data
        print("✓ All categories generated")

    with open("data/horoscopes.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n✅ All 12 rashis generated and saved successfully.")

if __name__ == "__main__":
    main()
