#!/usr/bin/env python3
"""
Factual daily horoscope generator using real planetary ephemeris.
Computes actual planet positions, then uses AI to interpret them per rashi.
"""

import json
import os
import time
import sys
import zipfile
import urllib.request
from datetime import datetime, timezone

import swisseph as swe   # pyswisseph

from sambanova import SambaNova

# ---------- EPHEMERIS SETUP (one-time download) ----------
EPHE_DIR = "ephe"
EPHE_URL = "https://github.com/aloistr/swisseph/raw/master/ephe/ephe.zip"

def ensure_ephemeris():
    """Download Swiss Ephemeris files if not already present."""
    if not os.path.exists(EPHE_DIR) or not os.listdir(EPHE_DIR):
        print("Downloading ephemeris files (one-time)...")
        os.makedirs(EPHE_DIR, exist_ok=True)
        zip_path = os.path.join(EPHE_DIR, "ephe.zip")
        urllib.request.urlretrieve(EPHE_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(EPHE_DIR)
        os.remove(zip_path)
        print("Ephemeris ready.")
    swe.set_ephe_path(EPHE_DIR)

# ---------- CONFIGURATION ----------
RASHIS = [
    "Mesha (Aries)", "Vrishabha (Taurus)", "Mithuna (Gemini)",
    "Karka (Cancer)", "Simha (Leo)", "Kanya (Virgo)",
    "Tula (Libra)", "Vrishchika (Scorpio)", "Dhanu (Sagittarius)",
    "Makara (Capricorn)", "Kumbha (Aquarius)", "Meena (Pisces)"
]

RASHI_NAMES_SHORT = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

# Planet list for horoscope
PLANETS = [
    swe.SUN, swe.MOON, swe.MARS, swe.MERCURY,
    swe.JUPITER, swe.VENUS, swe.SATURN,
    swe.MEAN_NODE   # Rahu – we'll take Ketu as opposite
]

PLANET_NAMES = {
    swe.SUN: "Sun", swe.MOON: "Moon", swe.MARS: "Mars",
    swe.MERCURY: "Mercury", swe.JUPITER: "Jupiter", swe.VENUS: "Venus",
    swe.SATURN: "Saturn", swe.MEAN_NODE: "Rahu"
}

API_KEY = os.environ["SAMBANOVA_API_KEY"]
BASE_URL = "https://api.sambanova.ai/v1"
client = SambaNova(api_key=API_KEY, base_url=BASE_URL)

MAX_OUTPUT_TOKENS = 1200
TEMPERATURE = 0.8
TOP_P = 0.9
RETRY_COUNT = 3
RETRY_DELAY = 2  # seconds

SYSTEM_PROMPT = (
    "You are a seasoned Vedic astrologer. "
    "You MUST base your entire reading ONLY on the factual planetary positions provided. "
    "Do not invent any placement not given. "
    "Return exactly the JSON structure requested. "
    "No markdown, no extra text. "
    "Keys: general, luck, scope, study, love, travel, lucky_number, lucky_color. "
    "All values strings except lucky_number (integer)."
)

# ---------- ASTROLOGICAL TOOLS ----------
def get_planet_data(jd):
    """Return dict of planet -> {sign_idx, degree, nakshatra} for a given Julian day."""
    data = {}
    for pid in PLANETS:
        lon, _ = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL | swe.FLG_SPEED)
        sign_idx = int(lon // 30)
        degree = lon % 30
        # Nakshatra (27 segments of 13°20')
        nakshatra_idx = int(lon // (360 / 27))
        data[pid] = {
            "sign_idx": sign_idx,
            "sign_name": RASHI_NAMES_SHORT[sign_idx],
            "degree": round(degree, 2),
            "nakshatra_idx": nakshatra_idx
        }
    # Ketu opposite Rahu
    rahu_sign = data[swe.MEAN_NODE]["sign_idx"]
    ketu_sign = (rahu_sign + 6) % 12
    data["Ketu"] = {
        "sign_idx": ketu_sign,
        "sign_name": RASHI_NAMES_SHORT[ketu_sign],
        "degree": (data[swe.MEAN_NODE]["degree"] + 180) % 360,
        "nakshatra_idx": (data[swe.MEAN_NODE]["nakshatra_idx"] + 13) % 27
    }
    return data

def build_transit_summary_for_rashi(rashi_idx, planet_data):
    """Create a string summarising house placements for the given rashi."""
    lines = []
    for pid, pdata in planet_data.items():
        planet_name = PLANET_NAMES.get(pid, pid)
        house = (pdata["sign_idx"] - rashi_idx) % 12 + 1
        lines.append(
            f"{planet_name} in house {house} ({pdata['sign_name']} {pdata['degree']}°)"
        )
    return " | ".join(lines)

# ---------- GENERATION ----------
def generate_rashi(rashi_idx, rashi_name, today_str, transit_text):
    """Generate full horoscope for one rashi using factual transit data."""
    user_prompt = (
        f"Rashi: {rashi_name}\nToday: {today_str}\n\n"
        f"Factual planetary positions (house placements from {rashi_name}):\n{transit_text}\n\n"
        f"Based ONLY on these exact positions, provide a detailed horoscope with these fields:\n"
        f"general, luck, scope, study, love, travel, lucky_number, lucky_color.\n"
        f"Return a JSON object. The horoscope text must be grounded in the given transits."
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
                raise ValueError("Empty response")

            # Parse JSON – handle markdown wrapping
            try:
                data = json.loads(content.strip())
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("JSON not found in response")

            # Validate
            required = {"general", "luck", "scope", "study", "love", "travel", "lucky_number", "lucky_color"}
            if not required.issubset(data):
                raise ValueError(f"Missing keys: {required - data.keys()}")

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
        f"Failed to generate horoscope for {rashi_name} after {RETRY_COUNT} attempts. "
        f"Last error: {last_error}"
    )

# ---------- MAIN ----------
def main():
    ensure_ephemeris()

    today = datetime.now()
    today_str = today.strftime("%B %d, %Y")
    today_iso = today.isoformat()
    jd = swe.julday(today.year, today.month, today.day, 0.0)

    # Compute global planet positions (sidereal)
    print("Computing planetary positions...")
    planet_data = get_planet_data(jd)
    print("Planetary positions:", {PLANET_NAMES.get(k, k): v["sign_name"] for k, v in planet_data.items()})

    os.makedirs("data", exist_ok=True)

    output = {
        "date": today_iso,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "rashi_horoscopes": {}
    }

    for idx, rashi in enumerate(RASHIS):
        print(f"\n===== {rashi} =====")
        transit_text = build_transit_summary_for_rashi(idx, planet_data)
        print(f"Transits: {transit_text[:200]}...")
        data = generate_rashi(idx, rashi, today_str, transit_text)
        output["rashi_horoscopes"][rashi] = data
        print("✓ Generated")

    with open("data/horoscopes.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n✅ All horoscopes saved with real astronomical accuracy.")

if __name__ == "__main__":
    main()
