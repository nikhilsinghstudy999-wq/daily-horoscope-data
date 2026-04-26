#!/usr/bin/env python3
"""
Production-grade, self-healing daily horoscope generator.
Uses Moshier built-in ephemeris (+ Lahiri ayanamsa) + SambaNova LLM.
Features a multi-stage JSON repair pipeline to handle any LLM output.
"""

import json
import os
import re
import sys
import time
import tempfile
from datetime import datetime, timezone

import swisseph as swe
from sambanova import SambaNova

# ──────────────────────────────────────────────
# 0. EPHEMERIS SETUP (BUILT-IN MOSHIER)
# ──────────────────────────────────────────────
def setup_ephemeris():
    """
    Initialize Swiss Ephemeris using Moshier built-in ephemeris.
    Requires NO external files. Sidereal positions are set with Lahiri ayanamsa.
    Returns the temporary directory path used for configuration.
    """
    ephe_dir = tempfile.mkdtemp(prefix="sweph_")
    swe.set_ephe_path(ephe_dir)
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    print(f"Ephemeris initialised (Moshier built-in, path={ephe_dir})")
    return ephe_dir

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

PLANETS = {
    swe.SUN:       "Sun",
    swe.MOON:      "Moon",
    swe.MARS:      "Mars",
    swe.MERCURY:   "Mercury",
    swe.JUPITER:   "Jupiter",
    swe.VENUS:     "Venus",
    swe.SATURN:    "Saturn",
    swe.MEAN_NODE: "Rahu",
}

# ──────────────────────────────────────────────
# 2. SAMBANOVA CLIENT
# ──────────────────────────────────────────────
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
if not SAMBANOVA_API_KEY:
    print("ERROR: SAMBANOVA_API_KEY environment variable is not set.")
    sys.exit(1)

client = SambaNova(api_key=SAMBANOVA_API_KEY, base_url="https://api.sambanova.ai/v1")

MODEL_NAME = "gpt-oss-120b"
MAX_OUTPUT_TOKENS = 1200
TEMPERATURE = 0.8
TOP_P = 0.9
RETRY_COUNT = 3
RETRY_DELAY = 3  # seconds

SYSTEM_PROMPT = (
    "You are a seasoned Vedic astrologer. "
    "You must respond with a single, raw JSON object. "
    "Do not use markdown formatting, line breaks, or extra text. "
    "The JSON must contain exactly these keys: general, luck, scope, study, love, travel, lucky_number, lucky_color. "
    "All values must be strings, except lucky_number which must be an integer. "
    "Make each text value 2-4 detailed sentences grounded in the given transits. "
    "Ensure the JSON is complete and not truncated."
)

# ──────────────────────────────────────────────
# 3. ASTRONOMICAL CALCULATIONS
# ──────────────────────────────────────────────
def compute_planet_positions(jd):
    """Calculate sidereal (Lahiri) longitudes for all planets at a given Julian day."""
    positions = {}
    for pid, pname in PLANETS.items():
        xx, ret_flag = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL | swe.FLG_SPEED)
        lon = xx[0]
        sign_idx = int(lon // 30)
        degree = round(lon % 30, 2)
        positions[pname] = {
            "sign_idx": sign_idx,
            "sign_name": RASHI_SHORT[sign_idx],
            "degree": degree,
        }
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
    """Format planet positions into a compact 'house placement' summary string."""
    parts = []
    for pname, data in positions.items():
        house = (data["sign_idx"] - rashi_idx) % 12 + 1
        parts.append(f"{pname} in house {house} ({data['sign_name']} {data['degree']}°)")
    return " | ".join(parts)

# ──────────────────────────────────────────────
# 4. ROBUST JSON PARSER (THE "SELF-HEALING" PART)
# ──────────────────────────────────────────────
def parse_and_validate_json(content):
    """
    Multi-stage JSON parsing designed for messy LLM output.
    Returns a parsed dictionary if successful, raises ValueError otherwise.
    """
    if not content:
        raise ValueError("Response content is empty.")

    # Stage 1: Direct parse (ideal case)
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    # Stage 2: Extract JSON object using regex (handles leading/trailing text)
    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    # Stage 3: Repair common structural issues (unbalanced braces, missing commas, quotes)
    try:
        repaired = repair_json_string(content)
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass

    # Stage 4: Aggressive repair + truncation fix (for truncated JSON)
    try:
        repaired = repair_truncated_json(content)
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass

    # Stage 5: Absolute last resort - try to complete the object by adding missing '}'
    try:
        cleaned_content = content.strip().rstrip(',')
        if cleaned_content.endswith('"') or cleaned_content.endswith(']'):
            cleaned_content += '}'
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"No JSON object found in response. First 200 chars: {content[:200]}")

def repair_json_string(json_str):
    """Fix common JSON formatting errors: unbalanced braces, missing commas, quotes."""
    # Remove any leading/trailing whitespace and markdown fences
    json_str = json_str.strip()
    json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
    json_str = re.sub(r'\s*```$', '', json_str)

    # Count braces
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    
    # Add missing closing braces if the string is truncated
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    # Remove extra closing braces
    elif close_braces > open_braces:
        json_str = json_str[::-1].replace('}', '', close_braces - open_braces)[::-1]

    # Fix missing commas between key-value pairs (common LLM mistake)
    json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
    # Replace single quotes with double quotes
    json_str = re.sub(r"(?<!\\)'", '"', json_str)
    
    return json_str

def repair_truncated_json(json_str):
    """Handle severely truncated JSON by smartly closing all structures."""
    json_str = json_str.strip()
    stack = []
    in_string = False
    escape_next = False
    
    for char in json_str:
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
        if in_string:
            continue
        if char in '{[':
            stack.append(char)
        elif char in '}]':
            if stack:
                stack.pop()
    
    # Close all unclosed structures
    closing = ''
    for char in reversed(stack):
        closing += '}' if char == '{' else ']'
    
    return json_str + closing

# ──────────────────────────────────────────────
# 5. LLM CALL WITH RETRIES
# ──────────────────────────────────────────────
def generate_rashi(rashi_name, today_str, transit_text):
    """Generate and validate a horoscope for one Rashi with retries."""
    user_prompt = (
        f"Rashi: {rashi_name}\nToday: {today_str}\n\n"
        f"Factual planetary positions (house placements from {rashi_name}):\n{transit_text}\n\n"
        f"Based ONLY on these exact positions, provide a detailed horoscope with these fields:\n"
        f"general, luck, scope, study, love, travel, lucky_number, lucky_color.\n"
        f"Return a JSON object. Ensure the JSON is complete and properly closed."
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
            data = parse_and_validate_json(content)

            # Validate content
            required = {"general", "luck", "scope", "study", "love", "travel",
                        "lucky_number", "lucky_color"}
            missing = required - data.keys()
            if missing:
                raise ValueError(f"Missing keys: {missing}")

            # Normalize types
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
# 6. MAIN
# ──────────────────────────────────────────────
def main():
    setup_ephemeris()

    today = datetime.now()
    today_str = today.strftime("%B %d, %Y")
    today_iso = today.isoformat()
    jd = swe.julday(today.year, today.month, today.day, 0.0)

    print("Computing planetary positions (Moshier ephemeris) …")
    positions = compute_planet_positions(jd)
    for name, data in positions.items():
        print(f"  {name}: {data['sign_name']} {data['degree']}°")

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

    out_path = os.path.join("data", "horoscopes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ All 12 rashis saved to {out_path}")

if __name__ == "__main__":
    main()
