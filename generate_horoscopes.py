#!/usr/bin/env python3
"""
Production-grade daily horoscope generator using Moshier built‑in ephemeris
and SambaNova LLM.  All planetary positions are astronomically correct.
"""

import json
import os
import sys
import time
import tempfile
from datetime import datetime, timezone

import swisseph as swe

from sambanova import SambaNova

# ──────────────────────────────────────────────
# 0. EPHEMERIS & AYANAMSA SETUP
# ──────────────────────────────────────────────
def setup_ephemeris():
    """
    Use the built‑in Moshier ephemeris (no download required).
    Also set Lahiri ayanamsa for sidereal positions.
    """
    # The library needs some path to exist; an empty temp dir triggers the
    # fallback to Moshier, which is more than accurate enough.
    ephe_dir = tempfile.mkdtemp(prefix="sweph_")
    swe.set_ephe_path(ephe_dir)
    # Sidereal mode with Lahiri ayanamsa (most common Vedic)
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    print(f"Ephemeris: Moshier built‑in, Lahiri ayanamsa (path={ephe_dir})")

# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────
RASHIS = [
    "Mesha (Aries)", "Vrishabha (Taurus)", "Mithuna (Gemini)",
    "Karka (Cancer)", "Simha (Leo)", "Kanya (Virgo)",
    "Tula (Libra)", "Vrishchika (Scorpio)", "Dhanu (Sagittarius)",
    "Makara (Capricorn)", "Kumbha (Aquarius)", "Meena (Pisces)"
]

RASHI_SHORT = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

# Planets we include in every horoscope
PLANETS = {
    swe.SUN:       "Sun",
    swe.MOON:      "Moon",
    swe.MARS:      "Mars",
    swe.MERCURY:   "Mercury",
    swe.JUPITER:   "Jupiter",
    swe.VENUS:     "Venus",
    swe.SATURN:    "Saturn",
    swe.MEAN_NODE: "Rahu",      # Ketu derived opposite
}

# ──────────────────────────────────────────────
# 2. SAMBANOVA CLIENT
# ──────────────────────────────────────────────
API_KEY = os.environ["SAMBANOVA_API_KEY"]
client = SambaNova(api_key=API_KEY, base_url="https://api.sambanova.ai/v1")

MODEL_NAME = "gpt-oss-120b"
MAX_OUTPUT_TOKENS = 1200
TEMPERATURE = 0.8
TOP_P = 0.9
RETRY_COUNT = 3
RETRY_DELAY = 3  # seconds

SYSTEM_PROMPT = (
    "You are a seasoned Vedic astrologer. "
    "You MUST base your entire reading ONLY on the factual planetary positions provided. "
    "Do not invent any placement not given. "
    "Return exactly the JSON structure requested. "
    "No markdown, no extra text, no code fences. "
    "The JSON keys are: general, luck, scope, study, love, travel, lucky_number, lucky_color. "
    "All values must be strings, except lucky_number which must be an integer. "
    "Make each text value 2‑4 detailed sentences grounded in the given transits."
)

# ──────────────────────────────────────────────
# 3. ASTRONOMICAL CALCULATIONS
# ──────────────────────────────────────────────
def compute_planet_positions(jd):
    """
    Calculate sidereal (Lahiri) longitudes for all planets at the given Julian day.

    Returns:
        dict: planet_name -> {"sign_idx": int, "sign_name": str, "degree": float}
    """
    positions = {}
    for pid, pname in PLANETS.items():
        # CORRECTED: Unpack the returned tuple properly.
        # The function returns a tuple where the first element is a list.
        # The first element of that list (xx[0]) is the longitude.
        xx, ret_flag = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL | swe.FLG_SPEED)
        lon = xx[0]          # longitude in degrees (0‑360)
        sign_idx = int(lon // 30)
        degree = round(lon % 30, 2)
        positions[pname] = {
            "sign_idx": sign_idx,
            "sign_name": RASHI_SHORT[sign_idx],
            "degree": degree,
        }
    # Ketu is exactly opposite Rahu
    rahu = positions["Rahu"]
    ketu_sign = (rahu["sign_idx"] + 6) % 12
    ketu_degree = (rahu["degree"] + 180) % 360
    positions["Ketu"] = {
        "sign_idx": ketu_sign,
        "sign_name": RASHI_SHORT[ketu_sign],
        "degree": round(ketu_degree, 2),
    }
    return positions

def build_transit_text(rashi_idx, positions):
    """
    Convert raw planet positions into a compact "house placement" summary
    for one rashi, e.g. "Sun in house 1 (Aries 5.3°) | Moon in house 4 (Cancer 12.1°) …"
    """
    parts = []
    for pname, data in positions.items():
        house = (data["sign_idx"] - rashi_idx) % 12 + 1
        parts.append(f"{pname} in house {house} ({data['sign_name']} {data['degree']}°)")
    return " | ".join(parts)

# ──────────────────────────────────────────────
# 4. LLM CALL WITH RETRIES
# ──────────────────────────────────────────────
def generate_rashi(rashi_name, today_str, transit_text):
    """Single API call → parsed & validated dict for one rashi."""
    user_prompt = (
        f"Rashi: {rashi_name}\nToday: {today_str}\n\n"
        f"Factual planetary positions (house placements from {rashi_name}):\n{transit_text}\n\n"
        f"Based ONLY on these exact positions, provide a detailed horoscope with these fields:\n"
        f"general, luck, scope, study, love, travel, lucky_number, lucky_color.\n"
        f"Return a JSON object."
    )

    last_error = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response content")

            # Parse JSON – handle occasional markdown wrapping
            try:
                data = json.loads(content.strip())
            except json.JSONDecodeError:
                import re
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if not match:
                    raise ValueError(f"No JSON object found in response: {content[:200]}")
                data = json.loads(match.group())

            # Validate required keys
            required = {"general", "luck", "scope", "study", "love", "travel",
                        "lucky_number", "lucky_color"}
            missing = required - data.keys()
            if missing:
                raise ValueError(f"Missing keys: {missing}")

            # Normalise types
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

# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────
def main():
    # --- Ephemeris ---
    setup_ephemeris()

    # --- Date ---
    today = datetime.now()
    today_str = today.strftime("%B %d, %Y")
    today_iso = today.isoformat()
    jd = swe.julday(today.year, today.month, today.day, 0.0)

    # --- Compute global planet positions ---
    print("Computing planetary positions (Moshier ephemeris) …")
    positions = compute_planet_positions(jd)
    for name, data in positions.items():
        print(f"  {name}: {data['sign_name']} {data['degree']}°")

    # --- Generate horoscopes ---
    os.makedirs("data", exist_ok=True)

    output = {
        "date": today_iso,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "rashi_horoscopes": {},
    }

    for idx, rashi in enumerate(RASHIS):
        print(f"\n===== {rashi} =====")
        transit = build_transit_text(idx, positions)
        print(f"Transits: {transit[:150]}…")
        data = generate_rashi(rashi, today_str, transit)
        output["rashi_horoscopes"][rashi] = data
        print("✓")

    # --- Write output ---
    out_path = os.path.join("data", "horoscopes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ All 12 rashis saved to {out_path}")

if __name__ == "__main__":
    main()
